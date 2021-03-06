import os
from contextlib import contextmanager
from io import BytesIO
from itertools import groupby, takewhile
from operator import attrgetter
from typing import (
    Any,
    BinaryIO,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import av

from .peekable import PeekableIterator

File = Union[str, BinaryIO]
FT = TypeVar("FT", bound=File)


@contextmanager
def resetting_offset(file: FT) -> Iterator[FT]:
    """Context manager for resetting the offset of a file to its initial value."""
    if hasattr(file, "seek"):
        offset = file.tell()
        try:
            yield file
        finally:
            file.seek(offset)
    else:
        yield file


@contextmanager
def copying_stream(
    src_file: File,
    dest_file: File,
    format: str = "mp4",
    stream_index: int = 0,
) -> Iterator[Tuple[av.container.OutputContainer, av.stream.Stream]]:
    """Context manager for adding a new stream from a source to a destination video file.

    :param src_file: Input file path or file-like object to copy
    :param dest_file: Output file path or file-like object
    :param format: Format of the output container
    :param stream_index: Index of the source stream channel to copy
    :return: (dest_container, dest_stream) tuple
    """
    with av.open(dest_file, mode="w", format=format) as out_container:
        with resetting_offset(src_file), av.open(src_file) as in_container:
            in_stream = in_container.streams[stream_index]
            out_stream = out_container.add_stream(template=in_stream)
        yield out_container, out_stream


def get_size_duration(
    file: File,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    stream_index: int = 0,
) -> Tuple[int, float]:
    """Get the size (in bytes) and duration (in seconds) of a video file stream.

    :param file: Video file to read
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :param stream_index: Index of the stream channel to read
    :return: (size, duration)
    """
    with resetting_offset(file), av.open(file) as container:
        packets = iter_packets_from_file(file, start_time, end_time, stream_index)
        size = sum(packet.buffer_size for packet in packets)
        # container.duration is in microsec
        duration = container.duration / 1e6
        return size, duration


def get_codec_context(file: File, stream_index: int = 0) -> Mapping[str, Any]:
    """Get a subset of a video stream codec context.

    :param file: Input video file path or file-like object
    :param stream_index: Index of the stream channel to inspect
    :return: Mapping of codec context properties
    """
    with av.open(file) as container:
        context = container.streams[stream_index].codec_context
        return {
            "codec": context.codec.name,
            "fps": context.framerate,
            "bitrate": context.bit_rate,
            "pixel_format": context.pix_fmt,
            "height": context.height,
            "width": context.width,
        }


def merge_files(
    src_files: Iterable[File],
    dest_file: File,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
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
    if not isinstance(src_files, PeekableIterator):
        src_files = PeekableIterator(src_files)
    if not src_files:
        return

    # set the output stream based on the first file
    with copying_stream(src_files.peek(), dest_file, format, stream_index) as (c, s):
        packets = PeekableIterator(
            iter_packets_from_files(src_files, start_time, end_time, stream_index)
        )
        try:
            first_dts = packets.peek().dts
        except StopIteration:
            return
        for packet in packets:
            # rewrite pts/dts of each packet so that first one in the file starts at zero
            packet.dts -= first_dts
            packet.pts -= first_dts
            packet.stream = s
            c.mux_one(packet)


def iter_frames_from_files(
    files: Iterable[File],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    stream_index: int = 0,
) -> Iterator[av.frame.Frame]:
    """Iterate over frames of zero or more video file streams.

    :param files: Video files to read
    :param start_time: Time offset of the first packet of the first file (in seconds)
    :param end_time: Time offset of the last packet of the last file (in seconds)
    :param stream_index: Index of the stream channel to read
    :return: Iterator of frames
    """
    # The obvious solution would be just:
    #
    #   for packet in iter_packets_from_files(files, start_time, end_time, stream_index):
    #       yield from packet.decode()
    #
    # Unfortunately this loses the first few frame(s) of each file, apparently because
    # they belong to different streams. What we need to do instead is to create a new
    # stream, reassign all packets of all files to it and then decode them.

    if not isinstance(files, PeekableIterator):
        files = PeekableIterator(files)
    if not files:
        return

    # set the output stream based on the first file
    # we need a container in order to create a stream but apparently the
    # container file is not really needed if all we do is reassign packets to
    # the stream and decode them, so just set the file to devnull
    with copying_stream(files.peek(), os.devnull, "mp4", stream_index) as (_, stream):
        for packet in iter_packets_from_files(
            files, start_time, end_time, stream_index
        ):
            packet.stream = stream
            yield from packet.decode()


def iter_frames_from_file(
    file: File,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    stream_index: int = 0,
) -> Iterator[av.frame.Frame]:
    """Iterate over frames of a video file stream.

    :param file: Video file to read
    :param start_time: Time offset of the first packet (in seconds)
    :param end_time: Time offset of the last packet (in seconds)
    :param stream_index: Index of the stream channel to read
    :return: Iterator of frames
    """
    for packet in iter_packets_from_file(file, start_time, end_time, stream_index):
        yield from packet.decode()


def iter_packets_from_files(
    files: Iterable[File],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
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
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
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
            packet for packet in container.demux(stream) if packet.dts is not None
        )

        if end_time is not None:
            # discard packets after end_time
            end_dts = int(end_time / stream.time_base)
            packets = takewhile(lambda p: p.dts <= end_dts, packets)

        yield from packets


def split_file(
    file: File,
    split_size: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    format: str = "mp4",
    stream_index: int = 0,
) -> Iterator[Tuple[float, float, bytes]]:
    """Split a video stream into smaller files.

    :param file: Video file to split
    :param split_size: Minimum size in bytes of each split file (with the possible
        exception of the last chunk)
    :param start_time: Start time offset (in seconds)
    :param end_time: End time offset (in seconds)
    :param format: Format of the split files
    :param stream_index: Index of the stream channel to split
    :return: Iterator of (start_time, end_time, bytes) tuples for each split file
    """
    packets = iter_packets_from_file(file, start_time, end_time, stream_index)
    for chunk in chunk_packets(packets, split_size):
        with resetting_offset(BytesIO()) as dest_file:
            with copying_stream(file, dest_file, format, stream_index) as (c, s):
                for packet in chunk:
                    packet.stream = s
                    c.mux_one(packet)

        time_breaks = [float(p.pts * p.time_base) for p in chunk]
        yield min(time_breaks), max(time_breaks), dest_file.read()


def chunk_packets(packets: Iterable[av.Packet], size: int) -> Iterator[List[av.Packet]]:
    """
    Split an iterable of packets into chunks of at least `size` bytes,
    with the possible exception of the last chunk.

    Yielded chunks are independent of each other, i.e. they start with a keyframe packet.

    :param packets: Iterable of packets
    :param size: Minimum chunk size in bytes
    :return: Iterator of packet lists
    """
    chunk = []
    chunk_bytes = 0
    # group packets by into consecutive keyframes and inbetweens
    for is_keyframe, group in groupby(packets, key=attrgetter("is_keyframe")):
        # add all the group packets into the current chunk
        for packet in group:
            if packet.buffer_size:
                chunk.append(packet)
                chunk_bytes += packet.buffer_size

        # yield a chunk when:
        # - it doesn't end in a keyframe (it may be needed for the following inbetweens)
        # - and its total size is at least `size`
        if not is_keyframe and chunk_bytes >= size:
            yield chunk
            chunk = []
            chunk_bytes = 0

    # yield the last chunk regardless of keyframe/size if non-empty
    if chunk:
        yield chunk
