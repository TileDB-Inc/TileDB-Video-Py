from contextlib import contextmanager
from typing import (
    Any,
    BinaryIO,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

File = Union[str, BinaryIO]
TimeOffset = Optional[float]

FT = TypeVar("FT", bound=File)


@contextmanager
def resetting_offset(file: FT) -> Iterator[FT]:
    """Context manager for resetting the offset of a file to its initial value."""
    if hasattr(file, "seek"):
        offset = file.tell()
        try:
            yield file
        finally:
            file.seek(offset)
    else:
        yield file


T = TypeVar("T")


class PeekableIterator(Generic[T]):
    """Wrap an iterator to allow lookahead.

    Simplified version of `more_itertools.peekable`:
         https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.peekable
    """

    _UNDEFINED = object()

    def __init__(self, iterable: Iterable[T]):
        self._it = iter(iterable)
        self._next: List[T] = []

    def __iter__(self) -> "PeekableIterator[T]":
        return self

    def __next__(self) -> T:
        try:
            return self._next.pop()
        except IndexError:
            return next(self._it)

    def __bool__(self) -> bool:
        try:
            self.peek()
        except StopIteration:
            return False
        return True

    def peek(self, default: Any = _UNDEFINED) -> Any:
        """Return the item that will be next returned from ``next()``.

        Return ``default`` if there are no items left. If ``default`` is not
        provided, raise ``StopIteration``.
        """
        try:
            return self._next[0]
        except IndexError:
            try:
                self._next.append(next(self._it))
            except StopIteration:
                if default is self._UNDEFINED:
                    raise
                return default
            else:
                return self.peek(default)
