from typing import Mapping, TypeVar
from collections.abc import Callable, Iterable, Iterator
from funcy import curry, lmap, remove

map_ = curry(map)
lmap_ = curry(lmap)
filter_ = curry(filter)
remove_ = curry(remove)

T = TypeVar('T')
def unique_if(pred: Callable[[T], object]) -> Callable[[Iterable[T]], Iterator[T]]:
    """Lazily removes duplicates of elements meeting the predicate."""
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

# Adapted from:
# https://github.com/Suor/funcy/blob/master/funcy/colls.py#L344-L348
def lwhere_not(**cond: object) -> Callable[[Iterable[Mapping]], list[Mapping]]:
    """Selects mappings omitting pairs in cond."""
    def _lwhere_not(mappings):
        return list(filter(
            lambda m: all(k in m and m[k] != v for k, v in cond.items()),
            mappings))

    return _lwhere_not

# Adapted from:
# https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#one
def exactly_one(iterable: Iterable[T], too_short=None, too_long=None) -> T:
    """Return the first item from *iterable*, which is expected to contain only
    that item. Raise an exception if *iterable* is empty or has more than one
    item."""
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
