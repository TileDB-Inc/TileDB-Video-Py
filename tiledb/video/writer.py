import pickle
from io import BytesIO
from itertools import groupby
from operator import attrgetter
from typing import Any, Iterable, Iterator, List, Mapping, Tuple

import av
import numpy as np

import tiledb

from .utils import File, resetting_offset


def from_file(
    uri: str,
    file: File,
    *,
    stream_index: int = 0,
    format: str = "mp4",
    split_duration: float = 1.0,
) -> None:
    """Create a TileDB array at given URI from a video file stream.

    :param uri: URI for new TileDB array
    :param file: Input video file path or file-like object
    :param stream_index: Index of the stream channel to write
    :param format: Format of the split video segments
    :param split_duration: Target duration of each split video segment (in seconds)
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
    size, duration = get_stream_size_duration(file, stream_index)
    split_size = int(size / duration * split_duration)

    with tiledb.open(uri, mode="w") as a:
        # split file and write each segment to tiledb
        for start, end, data in split_file(file, split_size, stream_index, format):
            a[start, end] = {"data": data, "size": len(data)}
        # write codec context as metadata
        a.meta["codec_context"] = pickle.dumps(get_codec_context(file, stream_index))


def get_stream_size_duration(file: File, stream_index: int = 0) -> Tuple[int, float]:
    """Get the size (in bytes) and duration (in seconds) of a video file stream.

    :param file: Video file to read
    :param stream_index: Index of the stream channel to read
    :return: (size, duration)
    """
    with resetting_offset(file):
        with av.open(file) as container:
            size = sum(packet.buffer_size for packet in container.demux(stream_index))
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


def split_file(
    file: File,
    size: int,
    stream_index: int = 0,
    format: str = "mp4",
) -> Iterator[Tuple[float, float, bytes]]:
    """Split a video stream into smaller files.

    :param file: Video file to split
    :param size: Minimum size in bytes of each split file (with the possible exception of
        the last chunk)
    :param stream_index: Index of the stream channel to split
    :param format: Format of the split files
    :return: Iterator of (start_time, end_time, bytes) tuples for each split file
    """
    with av.open(file) as in_container:
        in_stream = in_container.streams[stream_index]
        for chunk in chunk_packets(in_container.demux(in_stream), size):
            output_file = BytesIO()
            with av.open(output_file, "w", format=format) as out_container:
                out_stream = out_container.add_stream(template=in_stream)
                for packet in chunk:
                    # assign the packet to the new stream
                    packet.stream = out_stream
                    out_container.mux_one(packet)
            output_file.seek(0)
            time_breaks = [float(p.pts * p.time_base) for p in chunk]
            yield min(time_breaks), max(time_breaks), output_file.read()


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
