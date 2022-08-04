from typing import Mapping, TypeAlias, TypeVar
from collections.abc import Callable, Iterable, Iterator
import deal

T = TypeVar('T')
R = TypeVar('R')
UnaryPred: TypeAlias = Callable[[T], object]

@deal.pure
def lmap_(unary_op: Callable[[T], R]) -> Callable[[Iterable[T]], list[R]]:
    return lambda iterable: list(map(unary_op, iterable))

@deal.pure
def unique_if_(pred: UnaryPred[T]) -> Callable[[Iterable[T]], Iterator[T]]:
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
@deal.pure
def lwhere_not_(**cond: object) -> Callable[[Iterable[Mapping]], list[Mapping]]:
    """Selects mappings omitting pairs in cond."""
    return lambda mappings: list(filter(
        lambda m: all(k in m and m[k] != v for k, v in cond.items()),
        mappings))

# Adapted from:
# https://docs.python.org/3/library/itertools.html#itertools.dropwhile
@deal.pure
def dropwhile(pred: UnaryPred[T], iterable: Iterable[T]) -> Iterator[T]:

    iterable = iter(iterable)
    for x in iterable:
        if not pred(x):
            yield x
            break
    for x in iterable:
        yield x

# Adapted from:
# https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#rstrip
@deal.pure
def rstrip(iterable: Iterable[T], pred: UnaryPred[T]) -> Iterator[T]:

    cache = []
    for x in iterable:
        if pred(x):
            cache.append(x)
        else:
            yield from cache
            cache.clear()
            yield x

# Adapted from:
# https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#strip
@deal.pure
def strip(iterable: Iterable[T], pred: UnaryPred[T]) -> Iterator[T]:
    return rstrip(dropwhile(pred, iterable), pred)

@deal.pure
def strip_(pred: UnaryPred[T]) -> Callable[[Iterable[T]], Iterator[T]]:
    return lambda iterable: strip(iterable, pred)

# Adapted from:
# https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#one
UtException: TypeAlias = type[Exception] | Exception | None # type: ignore
@deal.pure
def one_expected(iterable: Iterable[T],
                 too_short: UtException=None,
                 too_long: UtException=None) -> T:
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
