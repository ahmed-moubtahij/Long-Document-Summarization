from abc import ABC, abstractmethod

import deal

class ExtractiveSummarizer(ABC):

    @abstractmethod
    @deal.pre(lambda _: _.n_sents > 0)
    def __call__(self, text: str, n_sents: int) -> str:
        ...
