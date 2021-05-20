import io
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from operator import attrgetter
from typing import Iterable, Iterator, List, Union

import av


@dataclass(frozen=True)
class StreamSegment:
    start_time: Fraction
    end_time: Fraction
    data: bytes


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


def split_stream(
    file: Union[str, BinaryIO],
    size: int,
    stream_index: int = 0,
    format: str = "mp4",
) -> Iterator[StreamSegment]:
    """Split a video stream into smaller files.

    :param file: Video file to split
    :param size: Minimum size in bytes of each split file (with the possible exception of
        the last chunk)
    :param stream_index: Index of the stream channel to split
    :param format: Format of the split files
    :return: Iterator of `StreamSegment` instances
    """
    with av.open(file) as in_container:
        in_stream = in_container.streams[stream_index]
        for chunk in chunk_packets(in_container.demux(in_stream), size):
            output_file = io.BytesIO()
            with av.open(output_file, "w", format=format) as out_container:
                out_stream = out_container.add_stream(template=in_stream)
                for packet in chunk:
                    # assign the packet to the new stream
                    packet.stream = out_stream
                    out_container.mux_one(packet)
            output_file.seek(0)
            time_breaks = [p.pts * p.time_base for p in chunk]
            yield StreamSegment(min(time_breaks), max(time_breaks), output_file.read())


def merge_files(
    src_files: Iterable[Union[str, io.IOBase]],
    dest_file: Union[str, io.IOBase],
    stream_index: int = 0,
    format: str = "mp4",
) -> None:
    """
    Merge a sequence of video files split by `split_stream`.

    :param src_files: File paths or file-like objects to merge
    :param dest_file: File path or file-like object to write the `src_files`
    :param stream_index: Index of the stream channel to read
    :param format: Format of the merged file.
    """
    out_stream = None
    with av.open(dest_file, "w", format=format) as out_container:
        for src_file in src_files:
            with av.open(src_file) as in_container:
                in_stream = in_container.streams[stream_index]
                if out_stream is None:
                    out_stream = out_container.add_stream(template=in_stream)

                for packet in in_container.demux(in_stream):
                    # We need to skip the "flushing" packets that `demux` generates.
                    if packet.dts is not None:
                        packet.stream = out_stream
                        out_container.mux_one(packet)


if __name__ == "__main__":
    import sys

    file = sys.argv[1]
    segment_files = []
    for i, segment in enumerate(split_stream(file, size=1024 ** 2)):
        print(
            f"Segment {i}: from {segment.start_time} to {segment.end_time} seconds "
            f"({len(segment.data)} bytes)"
        )
        segment_file = f"{file}-{i}"
        with open(segment_file, "wb") as f:
            f.write(segment.data)
        segment_files.append(segment_file)

    merge_files(segment_files, dest_file=f"{file}.merged")
