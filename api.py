from datetime import timedelta
from io import BytesIO
from typing import Iterator

import numpy as np
import tiledb

from split_merge import (
    File,
    TimeOffset,
    get_stream_size_duration,
    merge_files,
    split_stream,
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
        tiledb.Attr(name="duration", dtype=np.float64),
    ]
    schema = tiledb.ArraySchema(domain=domain, sparse=True, attrs=attrs)
    tiledb.Array.create(uri, schema)

    # determine the average size per split_interval
    size, duration = get_stream_size_duration(file, stream_index)
    split_size = int(size / duration * split_interval.total_seconds())

    with tiledb.open(uri, mode="w") as a:
        # segment input file and write each segment to tiledb
        for segment in split_stream(file, split_size, stream_index, format):
            a[segment.start_time, segment.end_time] = {
                "data": segment.data,
                "size": len(segment.data),
                "duration": segment.end_time - segment.start_time,
            }


def to_file(
    uri: str,
    file: File,
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
    src_files = list(_fetch_segment_files(uri, start_time, end_time))
    if src_files:
        merge_files(
            src_files=src_files,
            dest_file=file,
            format=format,
            start_time={src_files[0]: start_time},
            end_time={src_files[-1]: end_time},
        )


def _fetch_segment_files(
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
        print(a.query(attrs=["duration", "size"]).df[:])

    to_file(uri, f"{file}.merged", start_time=21, end_time=25)
