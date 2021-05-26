import pickle
from io import BytesIO
from itertools import takewhile
from typing import Any, Iterable, Iterator, Mapping, Optional, Union, cast

import av
from av.video.reformatter import Colorspace, Interpolation
from PIL.Image import Image

import tiledb

from .utils import File, PeekableIterator, TimeOffset, resetting_offset


def get_codec_context(uri: str) -> Mapping[str, Any]:
    """Get a subset of the video stream codec context.

    :param uri: URI of TileDB array to read from
    :return: Mapping of codec context properties
    """
    with tiledb.open(uri) as a:
        return cast(Mapping[str, Any], pickle.loads(a.meta["codec_context"]))


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
    segment_files = iter_segment_files(uri, start_time, end_time)
    merge_segment_files(segment_files, file, start_time, end_time, format)


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
    """
    Return an iterator of RGB images represented as `PIL.Image` instances from a TileDB array.

    :param uri: URI of TileDB array to read from
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :param width: New width, or None for the same width
    :param height: New height, or None for the same height
    :param src_colorspace: Current colorspace, or None for Colorspace.DEFAULT
    :param dst_colorspace: Desired colorspace, or None for Colorspace.DEFAULT
    :param interpolation: The interpolation method to use, or None for Interpolation.BILINEAR
    :return: Iterator of `PIL.Image` instances
    """
    segment_files = iter_segment_files(uri, start_time, end_time)
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
    :param format: New format, or None for the same format
    :param width: New width, or None for the same width
    :param height: New height, or None for the same height
    :param src_colorspace: Current colorspace, or None for Colorspace.DEFAULT
    :param dst_colorspace: Desired colorspace, or None for Colorspace.DEFAULT
    :param interpolation: The interpolation method to use, or None for Interpolation.BILINEAR
    :return: Iterator of `np.ndarray` instances
    """
    segment_files = iter_segment_files(uri, start_time, end_time)
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


def iter_segment_files(
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

    :param uri: URI of TileDB array to read from
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :return: Iterator of BytesIO buffers
    """
    # Get the segments overlapping with the (start_time, end_time) interval
    with tiledb.open(uri) as a:
        query = a.query(dims=[], attrs=["data"])
        chunks = query[slice(None, end_time), slice(start_time, None)]["data"]
    return map(BytesIO, chunks)


def merge_segment_files(
    src_files: Iterable[File],
    dest_file: File,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    format: str = "mp4",
    stream_index: int = 0,
) -> None:
    """
    Merge a sequence of video files split by `split_file`.

    :param src_files: File paths or file-like objects to merge
    :param dest_file: File path or file-like object to write the `src_files`
    :param start_time: Time offset of the first packet of the first src_file (in seconds)
    :param end_time: Time offset of the last packet of the last src_file (in seconds)
    :param format: Format of the merged file
    :param stream_index: Index of the stream channel to read
    """
    iter_src_files = PeekableIterator(src_files)
    if not iter_src_files:
        return

    with av.open(dest_file, "w", format=format) as out_container:
        # set the output stream with template taken from the first file stream
        with resetting_offset(iter_src_files.peek()) as first_src_file:
            with av.open(first_src_file) as in_container:
                in_stream = in_container.streams[stream_index]
                out_stream = out_container.add_stream(template=in_stream)

        iter_packets = PeekableIterator(
            iter_packets_from_files(iter_src_files, start_time, end_time, stream_index)
        )
        try:
            first_dts = iter_packets.peek().dts
        except StopIteration:
            return
        for packet in iter_packets:
            # rewrite pts/dts of each packet so that first one in the file starts at zero
            packet.dts -= first_dts
            packet.pts -= first_dts
            packet.stream = out_stream
            out_container.mux_one(packet)


def iter_packets_from_files(
    files: Iterable[File],
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    stream_index: int = 0,
) -> Iterator[av.Packet]:
    """Iterate over packets of zero or more video file streams.

    :param files: Video files to read
    :param start_time: Time offset of the first packet of the first file (in seconds)
    :param end_time: Time offset of the last packet of the last file (in seconds)
    :param stream_index: Index of the stream channel to read
    :return: Iterator of packets
    """
    if not isinstance(files, PeekableIterator):
        files = PeekableIterator(files)
    for i, src_file in enumerate(files):
        yield from iter_packets_from_file(
            src_file,
            start_time if i == 0 else None,
            end_time if not files else None,
            stream_index,
        )


def iter_packets_from_file(
    file: File,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    stream_index: int = 0,
) -> Iterator[av.Packet]:
    """Iterate over packets of a video file stream.

    :param file: Video file to read
    :param start_time: Time offset of the first packet (in seconds)
    :param end_time: Time offset of the last packet (in seconds)
    :param stream_index: Index of the stream channel to read
    :return: Iterator of packets
    """
    with av.open(file) as container:
        stream = container.streams[stream_index]

        if start_time is not None:
            file_end_time = (container.start_time + container.duration) / 1e6
            # if start_time is greater than the end of the file, don't yield any packets
            if start_time >= file_end_time:
                return
            # seek to the keyframe nearest to start_time
            container.seek(offset=int(start_time / stream.time_base), stream=stream)

        packets: Iterator[av.Packet] = (
            p for p in container.demux() if p.dts is not None
        )

        if end_time is not None:
            # discard packets after end_time
            end_dts = int(end_time / stream.time_base)
            packets = takewhile(lambda p: p.dts <= end_dts, packets)

        yield from packets
