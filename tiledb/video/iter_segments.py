from io import BytesIO
from typing import Iterator

import tiledb
from tiledb.video.utils import TimeOffset


def iter_segments(
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
