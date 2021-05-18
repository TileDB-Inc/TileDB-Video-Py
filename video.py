import os
import shutil
import itertools as it

import av
import numpy as np
import tiledb


def iter_chunks(iterable, n):
    iterator = iter(iterable)
    return iter(lambda: list(it.islice(iterator, n)), [])


def write_video(uri, path, stream_index=0, write_batch_tiles=10):
    with av.open(path) as container:
        stream = container.streams.video[stream_index]
        # for yuv420p/yuvj420p frame format we need
        # `h` values for Y + h/4 for U + h/4 for v == h * 3 / 2
        height = stream.height * 3 // 2
        width = stream.width
        frame_tile = round(stream.base_rate)
        domain = tiledb.Domain(
            tiledb.Dim(
                name="frame_id",
                domain=(0, stream.frames - 1),
                tile=frame_tile,
                dtype=np.uint32,
            ),
            tiledb.Dim(
                name="height", domain=(0, height - 1), tile=height, dtype=np.uint32
            ),
            tiledb.Dim(
                name="width", domain=(0, width - 1), tile=width, dtype=np.uint32
            ),
        )
        attr = tiledb.Attr(
            name="value",
            dtype=np.uint8,
            filters=tiledb.FilterList([tiledb.ZstdFilter()]),
        )
        schema = tiledb.ArraySchema(domain=domain, sparse=False, attrs=[attr])

        try:
            if os.path.exists(uri):
                shutil.rmtree(uri)
        except tiledb.TileDB:
            pass
        tiledb.Array.create(uri, schema)

        stream.thread_type = "AUTO"
        with tiledb.open(uri, mode="w") as tdb_array:
            num_batch_frames = write_batch_tiles * frame_tile
            iter_batch_frames = iter_chunks(container.decode(stream), num_batch_frames)
            offsets = range(0, stream.frames, num_batch_frames)
            for offset, batch_frames in zip(offsets, iter_batch_frames):
                frame_arrays = [f.to_ndarray(format="yuv420p") for f in batch_frames]
                tdb_array[offset : offset + len(batch_frames)] = np.stack(frame_arrays)
                # print(f"written {len(batch_frames)} frames from offset {offset}")


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1]
    tiledb_uri = video_path.replace(".mp4", ".tdb")
    write_video(tiledb_uri, video_path)
    with tiledb.open(tiledb_uri) as arr:
        print(arr.schema)
