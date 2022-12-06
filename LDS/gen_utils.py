from typing import Mapping, TypeAlias, TypeVar
from collections.abc import Callable, Iterable, Iterator, Hashable
import deal

T = TypeVar('T')
R = TypeVar('R')
H = TypeVar('H', bound=Hashable)
UnaryPred:      TypeAlias = Callable[[T], object]
UtException:    TypeAlias = type[Exception] | Exception | None # type: ignore[index]

@deal.pure
def identity_f(x: T) -> T:
    return x

@deal.has()
def stable_unique_list(seq: Iterable[H]) -> list[H]:
    return list(dict.fromkeys(seq))

@deal.pure
def exactly_one(*args: object) -> bool:
    """ Returns True if the sequence of arguments contains exactly one truthy value.
        Credits to Nahita from the Python discord.
    """
    it = iter(args)
    end_sentinel = object() # Can't be in `args` as it's unique per Python session
    first_true = next(filter(None, it), end_sentinel) # `None` acts as an Identity
    if first_true is end_sentinel:
        return False
    return next(filter(None, it), end_sentinel) is end_sentinel

@deal.has()
def reduce_(f: Callable[[T, T], T]) -> Callable[[Iterable[T]], T]:

    def _reduce(iterable):
        it = iter(iterable)
        v = next(it)
        for e in it:
            v = f(v, e)
        return v

    return _reduce

@deal.pure
def flat_map(
    anamorphism: Callable[[T], Iterable[R]],
    iterable: Iterable[T]
) -> Iterator[R]:

    for e in iterable:
        yield from anamorphism(e)

@deal.pure
def lmap_(unary_op: Callable[[T], R]) -> Callable[[Iterable[T]], list[R]]:
    return lambda iterable: list(map(unary_op, iterable))

@deal.pure
def unique_if_(pred: UnaryPred[H]) -> Callable[[Iterable[H]], Iterator[H]]:
    """Callable that lazily removes duplicates of elements meeting the predicate.
       Order is preserved.
    """
    def _unique_if(iterable):
        seen = set()
        for e in iterable:
            if pred(e):
                if not e in seen:
                    seen.add(e)
                    yield e
            else:
                yield e

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
@deal.has()
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
@deal.has()
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
@deal.has()
def strip(iterable: Iterable[T], pred: UnaryPred[T]) -> Iterator[T]:
    return rstrip(dropwhile(pred, iterable), pred)

@deal.has()
def strip_(pred: UnaryPred[T]) -> Callable[[Iterable[T]], Iterator[T]]:
    return lambda iterable: strip(iterable, pred)

# Adapted from:
# https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#one
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
