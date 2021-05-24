from contextlib import contextmanager
from typing import BinaryIO, Iterator, Mapping, Optional, Union

File = Union[str, BinaryIO]
TimeOffset = Optional[float]
FileTimeOffset = Union[TimeOffset, Mapping[File, TimeOffset]]


@contextmanager
def resetting_offset(file: File) -> Iterator[None]:
    """Context manager for resetting the offset of a file to its initial value."""
    if isinstance(file, str):
        yield
    else:
        offset = file.tell()
        try:
            yield
        finally:
            file.seek(offset)
