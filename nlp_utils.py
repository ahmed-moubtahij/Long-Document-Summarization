from collections.abc import Callable, Iterator
from functools import partial
import re

from spacy.lang.fr import French
import deal

@deal.pure
def trim(text: str) -> str:
    """Removes last sentence. Useful when the decoder generates it incompletely"""
    return '\n'.join(french_sentencizer(text)[:-1])

@deal.pure
def french_sentencizer(text: str) -> list[str]:
    # TODO: Verify whether `nlp` is cached, otherwise find a way to @cache it yourself
    nlp = French()
    nlp.add_pipe("sentencizer")

    return list(map(str, nlp(text).sents))


# TODO: Can the closure on `bisection` and `join_bisection` be @cache'd ?
deal.raises(ValueError, TypeError)
@deal.has()
def join_bisections() -> Callable[[str], Iterator[str]]:

    bisection = re.compile(r"(\w+)-\s(\w+)", flags=re.UNICODE)
    join_bisection = partial(bisection.sub, r"\1\2")

    return lambda text: map(join_bisection, text)
