import itertools as it
from contextlib import contextmanager
from io import BytesIO
from operator import attrgetter
from typing import BinaryIO, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import av

File = Union[str, BinaryIO]
TimeOffset = Optional[float]
FileTimeOffset = Union[TimeOffset, Mapping[File, TimeOffset]]


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
    for is_keyframe, group in it.groupby(packets, key=attrgetter("is_keyframe")):
        # add all the group packets into the current chunk
        for packet in group:
            # skip the "flushing" packets that `demux` generates
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


def split_file(
    file: File,
    size: int,
    stream_index: int = 0,
    fmt: str = "mp4",
) -> Iterator[Tuple[float, float, bytes]]:
    """Split a video stream into smaller files.

    :param file: Video file to split
    :param size: Minimum size in bytes of each split file (with the possible exception of
        the last chunk)
    :param stream_index: Index of the stream channel to split
    :param fmt: Format of the split files
    :return: Iterator of (start_time, end_time, bytes) tuples for each split file
    """
    with av.open(file) as in_container:
        in_stream = in_container.streams[stream_index]
        for chunk in chunk_packets(in_container.demux(in_stream), size):
            output_file = BytesIO()
            with av.open(output_file, "w", format=fmt) as out_container:
                out_stream = out_container.add_stream(template=in_stream)
                for packet in chunk:
                    # assign the packet to the new stream
                    packet.stream = out_stream
                    out_container.mux_one(packet)
            output_file.seek(0)
            time_breaks = [float(p.pts * p.time_base) for p in chunk]
            yield min(time_breaks), max(time_breaks), output_file.read()


@contextmanager
def resetting_offset(file: File) -> Iterator[None]:
    """Context manager for resetting the offset of a file to its initial value."""
    if isinstance(file, str):
        yield
    else:
        offset = file.tell()
        try:
            yield
        finally:
            file.seek(offset)


def get_stream_size_duration(file: File, stream_index: int = 0) -> Tuple[int, float]:
    """Get the size (in bytes) and duration (in seconds) of a video file stream.

    :param file: Video file to read
    :param stream_index: Index of the stream channel to read
    :return: (size, duration)
    """
    with resetting_offset(file), av.open(file) as container:
        size = sum(packet.buffer_size for packet in container.demux(stream_index))
        # container.duration is in microsec
        duration = container.duration / 1e6
        return size, duration


def iter_packets(
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
            packets = it.takewhile(lambda p: p.dts <= end_dts, packets)

        yield from packets


def merge_files(
    src_files: Iterable[File],
    dest_file: File,
    stream_index: int = 0,
    fmt: str = "mp4",
    start_time: FileTimeOffset = None,
    end_time: FileTimeOffset = None,
) -> None:
    """
    Merge a sequence of video files split by `split_file`.

    :param src_files: File paths or file-like objects to merge
    :param dest_file: File path or file-like object to write the `src_files`
    :param stream_index: Index of the stream channel to read
    :param fmt: Format of the merged file
    :param start_time: Start time offset (in seconds). It can be a mapping with files
        as keys and offsets as values, or a single offset for all files
    :param end_time: End time offset (in seconds). It can be a mapping with files
        as keys and offsets as values, or a single offset for all files
    """
    iter_src_files = iter(src_files)
    with av.open(dest_file, "w", format=fmt) as out_container:
        first_src_file = next(iter_src_files)
        with resetting_offset(first_src_file), av.open(first_src_file) as in_container:
            in_stream = in_container.streams[stream_index]
            out_stream = out_container.add_stream(template=in_stream)

        def get_time(fto: FileTimeOffset, f: File) -> TimeOffset:
            return fto.get(f) if isinstance(fto, Mapping) else fto

        packets = it.chain.from_iterable(
            iter_packets(
                src_file,
                stream_index,
                get_time(start_time, src_file),
                get_time(end_time, src_file),
            )
            for src_file in it.chain((first_src_file,), iter_src_files)
        )
        try:
            first_packet = next(packets)
        except StopIteration:
            return
        first_dts = first_packet.dts
        for packet in it.chain((first_packet,), packets):
            # rewrite pts/dts of each packet so that first one in the file starts at zero
            packet.dts -= first_dts
            packet.pts -= first_dts
            packet.stream = out_stream
            out_container.mux_one(packet)


if __name__ == "__main__":
    import sys

    file = sys.argv[1]
    segment_paths = []
    for i, (start, end, data) in enumerate(split_file(file, size=1024 ** 2)):
        print(f"Segment {i}: from {start} to {end} seconds " f"({len(data)} bytes)")
        segment_path = f"{file}-{i}"
        with open(segment_path, "wb") as f:
            f.write(data)
        segment_paths.append(segment_path)

    merge_files(segment_paths, dest_file=f"{file}.merged", start_time=21, end_time=25)
