__all__ = ["from_file", "to_file", "iter_images", "iter_ndarrays", "get_codec_context"]

import pickle
from typing import Any, Iterator, Mapping, Optional, Union, cast

import numpy as np
from av.video.reformatter import Colorspace, Interpolation
from PIL.Image import Image

import tiledb

from .iter_segments import iter_segments
from .utils import File, TimeOffset
from .utils import get_codec_context as get_codec_context_from_file
from .utils import get_size_duration, iter_packets_from_files, merge_files, split_file


def from_file(
    uri: str,
    file: File,
    *,
    split_duration: float = 1.0,
    format: str = "mp4",
    stream_index: int = 0,
) -> None:
    """Create a TileDB array at given URI from a video file stream.

    :param uri: URI for new TileDB array
    :param file: Input video file path or file-like object
    :param split_duration: Target duration of each split video segment (in seconds)
    :param format: Format of the split video segments
    :param stream_index: Index of the stream channel to write
    """
    # create schema
    domain = tiledb.Domain(
        *[
            tiledb.Dim(name, dtype=np.float64, domain=(0, 1e6), tile=1)
            for name in ("start_time", "end_time")
        ]
    )
    attrs = [
        tiledb.Attr(name="data", dtype=np.bytes_, filters=[tiledb.NoOpFilter()]),
        tiledb.Attr(name="size", dtype=np.uint32),
    ]
    schema = tiledb.ArraySchema(domain=domain, attrs=attrs, sparse=True)
    tiledb.Array.create(uri, schema)

    # determine the average size per split_interval
    size, duration = get_size_duration(file, stream_index)
    split_size = int(size / duration * split_duration)

    with tiledb.open(uri, mode="w") as a:
        # split file and write each segment to tiledb
        for start, end, data in split_file(file, split_size, format, stream_index):
            a[start, end] = {"data": data, "size": len(data)}
        # write codec context as metadata
        a.meta["codec_context"] = pickle.dumps(
            get_codec_context_from_file(file, stream_index)
        )


def to_file(
    uri: str,
    file: File,
    *,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    format: str = "mp4",
) -> None:
    """Read a video from a TileDB array into a file.

    :param uri: URI of TileDB array to read from
    :param file: Output video file path or file-like object
    :param format: Format of the output video file
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    """
    segment_files = iter_segments(uri, start_time, end_time)
    merge_files(segment_files, file, start_time, end_time, format)


def iter_images(
    uri: str,
    *,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    max_threads: int = 1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    src_colorspace: Union[Colorspace, str, None] = None,
    dst_colorspace: Union[Colorspace, str, None] = None,
    interpolation: Union[Interpolation, str, None] = None,
) -> Iterator[Image]:
    """
    Return an iterator of RGB images represented as `PIL.Image` instances from a TileDB array.

    :param uri: URI of TileDB array to read from
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :param max_threads: Max number of threads to fetch segments concurrently,
        or 0 to fetch all segments in a single call.
    :param width: New width, or None for the same width
    :param height: New height, or None for the same height
    :param src_colorspace: Current colorspace, or None for Colorspace.DEFAULT
    :param dst_colorspace: Desired colorspace, or None for Colorspace.DEFAULT
    :param interpolation: The interpolation method to use, or None for Interpolation.BILINEAR
    :return: Iterator of `PIL.Image` instances
    """
    segment_files = iter_segments(uri, start_time, end_time, max_threads)
    for packet in iter_packets_from_files(segment_files, start_time, end_time):
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
    max_threads: int = 1,
    format: str = "rgb24",
    width: Optional[int] = None,
    height: Optional[int] = None,
    src_colorspace: Union[Colorspace, str, None] = None,
    dst_colorspace: Union[Colorspace, str, None] = None,
    interpolation: Union[Interpolation, str, None] = None,
) -> Iterator[Image]:
    """Return an iterator of images represented as Numpy arrays from a TileDB array.

    :param uri: URI of TileDB array to read from
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :param max_threads: Max number of threads to fetch segments concurrently,
        or 0 to fetch all segments in a single call.
    :param format: New format, or None for the same format
    :param width: New width, or None for the same width
    :param height: New height, or None for the same height
    :param src_colorspace: Current colorspace, or None for Colorspace.DEFAULT
    :param dst_colorspace: Desired colorspace, or None for Colorspace.DEFAULT
    :param interpolation: The interpolation method to use, or None for Interpolation.BILINEAR
    :return: Iterator of `np.ndarray` instances
    """
    segment_files = iter_segments(uri, start_time, end_time, max_threads)
    for packet in iter_packets_from_files(segment_files, start_time, end_time):
        for frame in packet.decode():
            yield frame.to_ndarray(
                format=format,
                width=width,
                height=height,
                src_colorspace=src_colorspace,
                dst_colorspace=dst_colorspace,
                interpolation=interpolation,
            )


def get_codec_context(uri: str) -> Mapping[str, Any]:
    """Get a subset of the video stream codec context.

    :param uri: URI of TileDB array to read from
    :return: Mapping of codec context properties
    """
    with tiledb.open(uri) as a:
        return cast(Mapping[str, Any], pickle.loads(a.meta["codec_context"]))
