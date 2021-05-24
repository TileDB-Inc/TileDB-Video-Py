<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

# TileDB-Video-Py

TileDB-Video-Py is a Python library for writing a video stream as a
[TileDB array](https://docs.tiledb.com/main/basic-concepts/data-model) and slicing it efficiently.

## Installation

This project is currently in early development and can be installed directly from
[GitHub](https://github.com/TileDB-Inc/TileDB-Video-Py):

```bash
pip install git+https://github.com/TileDB-Inc/TileDB-Video-Py
```

## Usage

To see `tiledb.video` in action, you may review (or run locally) the included
[Jupyter notebook](https://github.com/TileDB-Inc/TileDB-Video-Py/examples/tiledb-video.ipynb).

## API

The API consists of the following functions under the `tiledb.video` namespace package:

### Writing to TileDB

- `from_file(uri, file, *, stream_index=0, format="mp4", split_duration=1)`

   Create a TileDB array at given URI from a video file stream.

    - `uri`: URI for new TileDB array
    - `file`: Input video file path or file-like object
    - `stream_index`: Index of the stream channel to write
    - `format`: Format of the split video segments
    - `split_duration`: Target duration of each split video segment (in seconds)


### Reading from TileDB

-  `to_file(uri, file, *, format="mp4", start_time=None, end_time=None)`

    Read a video from a TileDB array into a file.

    - `uri`: URI of TileDB array to read from
    - `file`: Output video file path or file-like object
    - `format`: Format of the output video file
    - `start_time`: Start time offset (in seconds)
    - `end_time`: End time offset (in seconds)

- `iter_images(uri, *, start_time=None, end_time=None, width=None, height=None,
               src_colorspace=None, dst_colorspace=None, interpolation=None)`

    Return an iterator of RGB images represented as
    [PIL.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image)
    instances from a TileDB array.

    - `uri`: URI of TileDB array to read from
    - `start_time`: Start time offset (in seconds)
    - `end_time`: End time offset (in seconds)

    For the remaining optional parameters, please consult the documentation of the PyAV
    [VideoReformatter.reformat()](https://pyav.org/docs/develop/api/video.html#av.video.reformatter.VideoReformatter.reformat)
    method.


- `iter_ndarrays(uri, *, start_time=None, end_time=None, format="rgb24", width=None,
                 height=None, src_colorspace=None, dst_colorspace=None, interpolation=None)`

    Return an iterator of images represented as
    [Numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
    from a TileDB array.

    - `uri`: URI of TileDB array to read from
    - `start_time`: Start time offset (in seconds)
    - `end_time`: End time offset (in seconds)

    For the remaining optional parameters, please consult the documentation of the PyAV
    [VideoReformatter.reformat()](https://pyav.org/docs/develop/api/video.html#av.video.reformatter.VideoReformatter.reformat)
    method.
