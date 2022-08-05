from collections.abc import Callable, Iterator
from functools import cache, partial
import re

from spacy.lang.fr import French
import deal

@deal.pure
def trim(text: str) -> str:
    """Removes last sentence. Useful when the decoder generates it incompletely"""
    return '\n'.join(french_sentencizer(text)[:-1])

@cache
def load_sentencizer() -> Callable:
    nlp = French()
    nlp.add_pipe("sentencizer")
    return nlp

@deal.pure
def french_sentencizer(text: str) -> list[str]:
    return list(map(str, load_sentencizer()(text).sents))

deal.raises(ValueError, TypeError)
@deal.has()
def join_bisections(text: str) -> Iterator[str]:
    # re.compile caches https://stackoverflow.com/a/12514276
    bisection = re.compile(r"(\w+)-\s(\w+)", flags=re.UNICODE)
    return map(partial(bisection.sub, r"\1\2"), text)
