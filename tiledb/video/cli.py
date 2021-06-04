"""Convert a video file to or from TileDB"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List, Optional

import av

from .api import from_file, to_file


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument("input", help="Input video file path or TileDB URI")
    parser.add_argument("output", help="Output video file path or TileDB URI")
    parser.add_argument("--from", type=float, help="Start time offset (in seconds)")
    parser.add_argument("--to", type=float, help="End time offset (in seconds)")
    parser.add_argument(
        "-f",
        "--format",
        default="mp4",
        help="Format of the output video file (for conversion from tileDB) "
        "or the split video segments (for conversion to tileDB)",
    )

    to_tiledb_args = parser.add_argument_group("Arguments for conversion to TileDB")
    to_tiledb_args.add_argument(
        "-l",
        "--segment-length",
        type=float,
        default=5.0,
        help="Split video segment length (in seconds)",
    )
    to_tiledb_args.add_argument(
        "-s",
        "--stream",
        type=int,
        default=0,
        help="Index of the stream channel to read",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = get_parser()
    args = parser.parse_args(argv)

    kwargs = dict(
        start_time=getattr(args, "from"),
        end_time=args.to,
        format=args.format,
    )
    try:
        with av.open(args.input):
            from_file(
                file=args.input,
                uri=args.output,
                split_duration=args.segment_length,
                stream_index=args.stream,
                **kwargs
            )
    except av.FFmpegError:
        to_file(uri=args.input, file=args.output, **kwargs)


if __name__ == "__main__":  # pragma: nocover
    main()
