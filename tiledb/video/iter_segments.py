from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Iterator

import tiledb
from tiledb.video.utils import TimeOffset


def iter_segments(
    uri: str,
    start_time: TimeOffset = None,
    end_time: TimeOffset = None,
    max_threads: int = 0,
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
    :param max_threads: Max number of threads to fetch segments concurrently,
        or 0 to fetch all segments in a single call.
    :return: Iterator of BytesIO buffers
    """
    # index to get the segments overlapping with the (start_time, end_time) interval
    idx = (slice(None, end_time), slice(start_time, None))
    with tiledb.open(uri) as a:
        query = a.query(dims=[], attrs=["data"])
        if not max_threads:
            yield from map(BytesIO, query[idx]["data"])
        else:

            def get_chunk_file(start: float, end: float, e: float = 1e-4) -> BytesIO:
                idx = (slice(max(start - e, 0), start + e), slice(end - e, end + e))
                chunks = query[idx]["data"]
                assert len(chunks) == 1
                return BytesIO(chunks[0])

            time_bounds = a.query(attrs=[])[idx]
            with ThreadPoolExecutor(max_workers=max_threads) as pool:
                yield from pool.map(
                    get_chunk_file, time_bounds["start_time"], time_bounds["end_time"]
                )
