import io
import pickle
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import tiledb
import tiledb.video as tv

VIDEO_PATH = Path(__file__).parent.parent / "examples" / "file_example_MP4_640_3MG.mp4"
assert VIDEO_PATH.is_file()


@pytest.fixture
def tiledb_uri(tmp_path):
    return str(tmp_path / VIDEO_PATH.with_suffix(".tdb").name)


@pytest.mark.parametrize("split_duration", [1, 5, 10, 20])
@pytest.mark.parametrize("start_time", [None, 7.1])
@pytest.mark.parametrize("end_time", [None, 19.5])
def test_from_file(tiledb_uri, split_duration, start_time, end_time):
    assert tiledb.object_type(tiledb_uri) is None
    tv.from_file(
        tiledb_uri,
        str(VIDEO_PATH),
        split_duration=split_duration,
        start_time=start_time,
        end_time=end_time,
    )
    assert tiledb.object_type(tiledb_uri) == "array"

    # white-box testing
    with tiledb.open(tiledb_uri) as a:
        schema = a.schema
        assert schema.domain.ndim == 2
        assert schema.domain.dim(0).name == "start_time"
        assert schema.domain.dim(0).dtype == np.dtype("float64")
        assert schema.domain.dim(1).name == "end_time"
        assert schema.domain.dim(1).dtype == np.dtype("float64")

        assert schema.nattr == 2
        assert schema.attr(0).name == "data"
        assert schema.attr(0).dtype == np.dtype("S")
        assert schema.attr(1).name == "size"
        assert schema.attr(1).dtype == np.dtype("uint32")

        assert set(a.meta.keys()) == {"codec_context"}
        codec_context = pickle.loads(a.meta["codec_context"])
        assert isinstance(codec_context, dict)
        assert set(codec_context.keys()) == {
            "codec",
            "fps",
            "bitrate",
            "pixel_format",
            "height",
            "width",
        }


@pytest.mark.parametrize("split_duration", [1, 5, 10, 20])
@pytest.mark.parametrize("start_time", [None, 7.1])
@pytest.mark.parametrize("end_time", [None, 19.5])
def test_to_file(tiledb_uri, tmp_path, split_duration, start_time, end_time):
    tv.from_file(tiledb_uri, str(VIDEO_PATH), split_duration=split_duration)

    # write to file and assert it is a valid MP4 file
    dest_file = str(tmp_path / VIDEO_PATH.name)
    tv.to_file(tiledb_uri, dest_file, start_time=start_time, end_time=end_time)
    file_type_info = subprocess.check_output(["file", dest_file]).decode()
    assert "ISO Media, MP4" in file_type_info and "IS0 14496" in file_type_info

    # write to in-memory buf and assert it has the same contents as the file
    dest_buf = io.BytesIO()
    tv.to_file(tiledb_uri, dest_buf, start_time=start_time, end_time=end_time)
    with open(dest_file, "rb") as f:
        assert f.read() == dest_buf.getvalue()


def test_get_codec_context(tiledb_uri):
    tv.from_file(tiledb_uri, str(VIDEO_PATH))

    tiledb_codec_context = tv.get_codec_context(tiledb_uri)
    assert tiledb_codec_context == {
        "codec": "h264",
        "fps": Fraction(30, 1),
        "bitrate": 710666,
        "pixel_format": "yuv420p",
        "height": 360,
        "width": 640,
    }

    file_codec_context = tv.utils.get_codec_context(str(VIDEO_PATH))
    assert file_codec_context == tiledb_codec_context


@pytest.mark.parametrize("split_duration", [1, 5, 10, 20])
@pytest.mark.parametrize("start_time", [None, 7.1])
@pytest.mark.parametrize("end_time", [None, 19.5])
def test_iter_ndarrays(tiledb_uri, split_duration, start_time, end_time):
    tv.from_file(tiledb_uri, str(VIDEO_PATH), split_duration=split_duration)

    kwargs = dict(start_time=start_time, end_time=end_time)
    file_ndarrays = list(
        frame.to_ndarray(format="rgb24")
        for frame in tv.utils.iter_frames_from_file(str(VIDEO_PATH), **kwargs)
    )
    tiledb_ndarrays = list(tv.iter_ndarrays(tiledb_uri, **kwargs))

    assert len(file_ndarrays) == len(tiledb_ndarrays)
    for a, b in zip(file_ndarrays, tiledb_ndarrays):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("split_duration", [1, 5, 10, 20])
@pytest.mark.parametrize("start_time", [None, 7.1])
@pytest.mark.parametrize("end_time", [None, 19.5])
def test_iter_images(tiledb_uri, split_duration, start_time, end_time):
    tv.from_file(tiledb_uri, str(VIDEO_PATH), split_duration=split_duration)

    kwargs = dict(start_time=start_time, end_time=end_time)
    file_images = list(
        frame.to_image()
        for frame in tv.utils.iter_frames_from_file(str(VIDEO_PATH), **kwargs)
    )
    tiledb_images = list(tv.iter_images(tiledb_uri, **kwargs))

    assert file_images == tiledb_images
