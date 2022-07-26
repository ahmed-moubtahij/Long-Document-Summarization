from typing import TypeVar
from collections.abc import Callable, Iterable, Iterator

map_: Callable = lambda f: lambda iterable: map(f, iterable)
filter_: Callable = lambda p: lambda iterable: filter(p, iterable)

T = TypeVar('T')
def unique_if(pred: Callable[[T], object]) -> Callable[[Iterable[T]], Iterator[T]]:

    def _unique_if(iterable):
        seen = set()
        for item in iterable:
            if pred(item):
                if not item in seen:
                    seen.add(item)
                    yield item
            else:
                yield item

    return _unique_if

# https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#one
def exactly_one(iterable: Iterable[T], too_short=None, too_long=None) -> T:

    it_ = iter(iterable)

    try:
        first_value = next(it_)
    except StopIteration as exc:
        raise (
            too_short or ValueError('too few items in iterable (expected 1)')
        ) from exc

    try:
        second_value = next(it_)
    except StopIteration:
        pass
    else:
        msg = ("Expected exactly one item in iterable,"
               f"but got {first_value!r}, {second_value!r}, and perhaps more.")
        raise too_long or ValueError(msg)

    return first_value
