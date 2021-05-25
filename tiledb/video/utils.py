from contextlib import contextmanager
from typing import BinaryIO, Iterator, Mapping, Optional, TypeVar, Union

File = Union[str, BinaryIO]
TimeOffset = Optional[float]
FileTimeOffset = Union[TimeOffset, Mapping[File, TimeOffset]]

T = TypeVar("T", bound=File)


@contextmanager
def resetting_offset(file: T) -> Iterator[T]:
    """Context manager for resetting the offset of a file to its initial value."""
    if hasattr(file, "seek"):
        offset = file.tell()
        try:
            yield file
        finally:
            file.seek(offset)
    else:
        yield file
