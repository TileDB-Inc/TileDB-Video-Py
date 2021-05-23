from datetime import timedelta
from io import BytesIO
from typing import Iterator, Optional, Union

import numpy as np
import tiledb
from av.packet import Packet
from av.video.reformatter import Colorspace, Interpolation
from PIL.Image import Image

from split_merge import (
    File,
    TimeOffset,
    get_stream_size_duration,
    iter_packets,
    merge_files,
    split_file,
)


def from_file(
    uri: str,
    file: File,
    *,
    stream_index: int = 0,
    format: str = "mp4",
    split_interval: timedelta = timedelta(seconds=1),
) -> None:
    """Create TileDB array at given URI from a video file stream.

    :param uri: URI for new TileDB array
    :param file: Input video file
    :param stream_index: Index of the stream channel to ingest
    :param format: Format of the split video segments
    :param split_interval: Target duration of each split video segment
    """
    # create schema
    max_duration = timedelta(days=365).total_seconds()
    domain = tiledb.Domain(
        *[
            tiledb.Dim(name, dtype=np.float64, domain=(0, max_duration), tile=1)
            for name in ("start_time", "end_time")
        ]
    )
    attrs = [
        tiledb.Attr(name="data", dtype=np.bytes_, filters=[tiledb.NoOpFilter()]),
        tiledb.Attr(name="size", dtype=np.uint32),
    ]
    schema = tiledb.ArraySchema(domain=domain, sparse=True, attrs=attrs)
    tiledb.Array.create(uri, schema)

    # determine the average size per split_interval
    size, duration = get_stream_size_duration(file, stream_index)
    split_size = int(size / duration * split_interval.total_seconds())

    with tiledb.open(uri, mode="w") as a:
        # segment input file and write each segment to tiledb
        for start, end, data in split_file(file, split_size, stream_index, format):
            a[start, end] = {"data": data, "size": len(data)}


def to_file(
    uri: str,
    file: File,
    *,
    format: str = "mp4",
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
) -> None:
    """Read a video from a TileDB array into a file.

    :param uri: URI for new TileDB array
    :param file: Output video file
    :param format: Format of the output file
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    """
    src_files = list(_iter_segment_files(uri, start_time, end_time))
    if src_files:
        merge_files(
            src_files,
            dest_file=file,
            format=format,
            start_time={src_files[0]: start_time},
            end_time={src_files[-1]: end_time},
        )


def iter_images(
    uri: str,
    *,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    src_colorspace: Union[Colorspace, str, None] = None,
    dst_colorspace: Union[Colorspace, str, None] = None,
    interpolation: Union[Interpolation, str, None] = None,
) -> Iterator[Image]:
    for packet in _iter_packets(uri, start_time, end_time):
        for frame in packet.decode():
            yield frame.to_image(
                width=width,
                height=height,
                src_colorspace=src_colorspace,
                dst_colorspace=dst_colorspace,
                interpolation=interpolation,
            )


def iter_ndarrays(
    uri: str,
    *,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    format: str = "rgb24",
    width: Optional[int] = None,
    height: Optional[int] = None,
    src_colorspace: Union[Colorspace, str, None] = None,
    dst_colorspace: Union[Colorspace, str, None] = None,
    interpolation: Union[Interpolation, str, None] = None,
) -> Iterator[Image]:
    for packet in _iter_packets(uri, start_time, end_time):
        for frame in packet.decode():
            yield frame.to_ndarray(
                format=format,
                width=width,
                height=height,
                src_colorspace=src_colorspace,
                dst_colorspace=dst_colorspace,
                interpolation=interpolation,
            )


def _iter_packets(
    uri: str, start_time: TimeOffset = None, end_time: TimeOffset = None
) -> Iterator[Packet]:
    src_files = list(_iter_segment_files(uri, start_time, end_time))
    if src_files:
        # filter packets by start_time from the first file and by end_time from the last
        # yield all packets from the intermediate files
        start_time_mapping = {src_files[0]: start_time}
        end_time_mapping = {src_files[-1]: end_time}
        for src_file in src_files:
            yield from iter_packets(
                src_file,
                start_time=start_time_mapping.get(src_file),
                end_time=end_time_mapping.get(src_file),
            )


def _iter_segment_files(
    uri: str, start_time: TimeOffset = None, end_time: TimeOffset = None
) -> Iterator[BytesIO]:
    """Fetch the minimum sequence of segment files that cover the given time range.

    # A segment s is overlapping if s.start_time <= end_time AND s.end_time >= start_time
    #
    # |----A---|-------B------|-----C---|------D----|----E----|-----F-----|
    #              x                                  y
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #              >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #
    # Overlapping segments: B, C, D, E

    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :return: Iterator of BytesIO buffers
    """
    # Get the segments overlapping with the (start_time, end_time) interval
    with tiledb.open(uri) as a:
        query = a.query(dims=[], attrs=["data"])
        chunks = query[slice(None, end_time), slice(start_time, None)]["data"]
    return map(BytesIO, chunks)


if __name__ == "__main__":
    import os
    import shutil
    import sys

    file = sys.argv[1]
    uri = sys.argv[1] + ".tdb"

    if os.path.isdir(uri):
        shutil.rmtree(uri)
    from_file(uri, file, split_interval=timedelta(seconds=5))

    with tiledb.open(uri) as a:
        print(a.schema)
        print(a.query(attrs=["size"]).df[:])

    start_time, end_time = 21, 25
    to_file(uri, f"{file}.merged", start_time=start_time, end_time=end_time)

    images = list(iter_images(uri, start_time=start_time, end_time=end_time))
    print(f"{len(images)} images from {start_time} to {end_time} second")

    ndarrays = np.stack(
        list(iter_ndarrays(uri, start_time=start_time, end_time=end_time))
    )
    print(f"ndarray shape from {start_time} to {end_time} second: {ndarrays.shape}")
