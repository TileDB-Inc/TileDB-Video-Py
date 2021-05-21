from datetime import timedelta
from typing import BinaryIO, Union

import numpy as np
import tiledb

from split_merge import get_stream_size_duration, split_stream


def from_file(
    uri: str,
    file: Union[str, BinaryIO],
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

    # segment input file and write each segment to tiledb
    with tiledb.open(uri, mode="w") as a:
        for segment in split_stream(file, split_size, stream_index, format):
            a[segment.start_time, segment.end_time] = {
                "data": segment.data,
                "size": len(segment.data),
                "duration": segment.end_time - segment.start_time,
            }


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
