import pickle
from io import BytesIO
from itertools import chain, takewhile
from typing import Any, Iterable, Iterator, Mapping, Optional, Union, cast

import av
from av.video.reformatter import Colorspace, Interpolation
from PIL.Image import Image

import tiledb

from .utils import File, FileTimeOffset, PeekableIterator, TimeOffset, resetting_offset


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
    format: str = "mp4",
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
) -> None:
    """Read a video from a TileDB array into a file.

    :param uri: URI of TileDB array to read from
    :param file: Output video file path or file-like object
    :param format: Format of the output video file
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    """
    src_files = list(iter_segment_files(uri, start_time, end_time))
    if src_files:
        merge_segment_files(
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
    for packet in iter_packets_from_tiledb(uri, start_time, end_time):
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
    for packet in iter_packets_from_tiledb(uri, start_time, end_time):
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
    stream_index: int = 0,
    format: str = "mp4",
    start_time: FileTimeOffset = None,
    end_time: FileTimeOffset = None,
) -> None:
    """
    Merge a sequence of video files split by `split_file`.

    :param src_files: File paths or file-like objects to merge
    :param dest_file: File path or file-like object to write the `src_files`
    :param stream_index: Index of the stream channel to read
    :param format: Format of the merged file
    :param start_time: Start time offset (in seconds). It can be a mapping with files
        as keys and offsets as values, or a single offset for all files
    :param end_time: End time offset (in seconds). It can be a mapping with files
        as keys and offsets as values, or a single offset for all files
    """
    iter_src_files = iter(src_files)
    with av.open(dest_file, "w", format=format) as out_container:
        with resetting_offset(next(iter_src_files)) as first_src_file:
            with av.open(first_src_file) as in_container:
                in_stream = in_container.streams[stream_index]
                out_stream = out_container.add_stream(template=in_stream)

        def get_time(fto: FileTimeOffset, f: File) -> TimeOffset:
            return fto.get(f) if isinstance(fto, Mapping) else fto

        packets = chain.from_iterable(
            iter_packets_from_file(
                src_file,
                stream_index,
                get_time(start_time, src_file),
                get_time(end_time, src_file),
            )
            for src_file in chain((first_src_file,), iter_src_files)
        )
        try:
            first_packet = next(packets)
        except StopIteration:
            return
        first_dts = first_packet.dts
        for packet in chain((first_packet,), packets):
            # rewrite pts/dts of each packet so that first one in the file starts at zero
            packet.dts -= first_dts
            packet.pts -= first_dts
            packet.stream = out_stream
            out_container.mux_one(packet)


def iter_packets_from_file(
    file: File,
    stream_index: int = 0,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
) -> Iterator[av.Packet]:
    """Iterate over packets of a video file stream.

    :param file: Video file to read
    :param stream_index: Index of the stream channel to read
    :param start_time: Time offset in seconds of the first frame
    :param end_time: Time offset in seconds of the last frame
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


def iter_packets_from_tiledb(
    uri: str, start_time: TimeOffset = None, end_time: TimeOffset = None
) -> Iterator[av.Packet]:
    """
    Return an iterator of encoded data packets from a TileDB array.

    :param uri: URI of TileDB array to read from
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :return: Iterator of `av.Packet` instances
    """
    iter_src_files = PeekableIterator(iter_segment_files(uri, start_time, end_time))
    # filter packets by start_time from the first file and by end_time from the last
    # yield all packets from the intermediate files
    for i, src_file in enumerate(iter_src_files):
        yield from iter_packets_from_file(
            src_file,
            start_time=start_time if i == 0 else None,
            end_time=end_time if not iter_src_files else None,
        )
