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

To see `tiledb.video` in action, you may view or run locally the
[Jupyter notebook demo](https://github.com/TileDB-Inc/TileDB-Video-Py/examples/tiledb-video.ipynb).

## API

The API consists of the following functions under the `tiledb.video` namespace package:

### Writing to TileDB

- `from_file(uri, file, *, split_duration=1.0, start_time=None, end_time=None, format="mp4", stream_index=0)`

   Create a TileDB array at given URI from a video file stream.

    - `uri`: URI for new TileDB array
    - `file`: Input video file path or file-like object
    - `split_duration`: Target duration of each split video segment (in seconds)
    - `start_time`: Start time offset (in seconds)
    - `end_time`: End time offset (in seconds)
    - `format`: Format of the split video segments
    - `stream_index`: Index of the stream channel to write


### Reading from TileDB

- `get_codec_context(uri)`

    Get a subset of the video stream codec context attributes.

    - `uri`: URI of TileDB array to read from

- `to_file(uri, file, *, format="mp4", start_time=None, end_time=None)`

    Read a video from a TileDB array into a file.

    - `uri`: URI of TileDB array to read from
    - `file`: Output video file path or file-like object
    - `start_time`: Start time offset (in seconds)
    - `end_time`: End time offset (in seconds)
    - `format`: Format of the output video file

- `iter_images(uri, *, start_time=None, end_time=None, max_threads=1, width=None,
               height=None, src_colorspace=None, dst_colorspace=None, interpolation=None)`

    Return an iterator of RGB images represented as
    [PIL.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image)
    instances from a TileDB array.

    - `uri`: URI of TileDB array to read from
    - `start_time`: Start time offset (in seconds)
    - `end_time`: End time offset (in seconds)
    - `max_threads`: Max number of threads to fetch segments concurrently, or 0 to fetch
      all segments in a single call

    For the remaining optional parameters, please consult the documentation of the PyAV
    [VideoReformatter.reformat()](https://pyav.org/docs/develop/api/video.html#av.video.reformatter.VideoReformatter.reformat)
    method.


- `iter_ndarrays(uri, *, start_time=None, end_time=None, max_threads=1, format="rgb24",
                 width=None, height=None, src_colorspace=None, dst_colorspace=None,
                 interpolation=None)`

    Return an iterator of images represented as
    [Numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
    from a TileDB array.

    - `uri`: URI of TileDB array to read from
    - `start_time`: Start time offset (in seconds)
    - `end_time`: End time offset (in seconds)
    - `max_threads`: Max number of threads to fetch segments concurrently, or 0 to fetch
      all segments in a single call

    For the remaining optional parameters, please consult the documentation of the PyAV
    [VideoReformatter.reformat()](https://pyav.org/docs/develop/api/video.html#av.video.reformatter.VideoReformatter.reformat)
    method.


## CLI

A simple `tiledb-video` command line interface is also provided for converting between
video files and TileDB arrays:

    $ tiledb-video --help
    usage: tiledb-video [-h] [--from FROM] [--to TO] [-f FORMAT]
                        [-l SEGMENT_LENGTH] [-s STREAM]
                        input output

    Convert a video file to or from TileDB

    positional arguments:
      input                 Input video file path or TileDB URI
      output                Output video file path or TileDB URI

    optional arguments:
      -h, --help            show this help message and exit
      --from FROM           Start time offset (in seconds) (default: None)
      --to TO               End time offset (in seconds) (default: None)
      -f FORMAT, --format FORMAT
                            Format of the output video file (for conversion from
                            tileDB) or the split video segments (for conversion to
                            tileDB) (default: mp4)

    Arguments for conversion to TileDB:
      -l SEGMENT_LENGTH, --segment-length SEGMENT_LENGTH
                            Split video segment length (in seconds) (default: 5.0)
      -s STREAM, --stream STREAM
                            Index of the stream channel to read (default: 0)
