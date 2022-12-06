from typing import ClassVar
import random

import deal

from LDS.ext_summarizers import ExtractiveSummarizer
from LDS.nlp_utils import french_sentencizer

class RandomSum(ExtractiveSummarizer):
    seeded: ClassVar = random.Random(42)

    deal.has('random')
    @deal.ensure(lambda _: len(_.result) <= len(_.text))
    def __call__(self, text: str, n_sents: int, joiner='\n') -> str:

        super().__call__(text, n_sents)
        sents = french_sentencizer(text)
        rand_sents = self.__class__.seeded.sample(sents, n_sents)

        return joiner.join(rand_sents).replace('\n\n', '\n')
