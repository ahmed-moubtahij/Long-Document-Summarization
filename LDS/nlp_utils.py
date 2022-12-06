from collections.abc import Callable
from typing import Protocol
from functools import cache
import re

from spacy.lang.fr import French
from numpy.typing import NDArray

import deal

RE_ALPHA = r'(?![0-9_])\w'
""" Alphabetic characters.
Refuses `[0-9_]` from the next match with negative lookahead, then applies \\w.
https://docs.python.org/3/library/re.html#index-32
For Unicode (str) patterns:
Matches Unicode word characters; this includes most characters that can
be part of a word in any language, as well as numbers and the underscore.
"""
match_anything     : re.Pattern[str] = re.compile(".*", flags=re.DOTALL)
match_nothing      : re.Pattern[str] = re.compile("a^")

class SentenceEncoderProto(Protocol):

    def encode(self, sentences: str | list[str]) -> NDArray:
        ...

@cache
def load_sentencizer() -> Callable:
    nlp = French()
    nlp.add_pipe("sentencizer")
    return nlp

def french_sentencizer(text: str) -> list[str]:
    return list(map(str, load_sentencizer()(text).sents))

@deal.pure
def trim(text: str) -> str:
    """Removes last sentence. Useful as it tends to be incomplete in decoding."""
    return '\n'.join(french_sentencizer(text)[:-1])
