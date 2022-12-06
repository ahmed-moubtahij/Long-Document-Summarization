from collections.abc import Callable
from typing import Literal

import deal

from LDS.ext_summarizers import ExtractiveSummarizer
from LDS.randomsum import RandomSum
from LDS.textrank import TextRank

from LDS.abs_summarizers import AbstractiveSummarizer
from LDS.abs_summarizers import CamembertSum
from LDS.abs_summarizers import MbartSum

from LDS.nlp_utils import french_sentencizer

ModelName = Literal["camembertsum", "mbartsum", "randomsum", "textrank"]

@deal.raises(NotImplementedError, ImportError, ValueError, TypeError)
@deal.has('io')
def summarizer_factory(
    model_name: ModelName,
    sentence_encoder: TextRank.SentenceEncoder = "camembert",
    n_tokens=512
) -> tuple[AbstractiveSummarizer | ExtractiveSummarizer, Callable]:

    match model_name:
        case "camembertsum":
            return CamembertSum(), lambda _, n_tokens=n_tokens: n_tokens

        case "mbartsum":
            return MbartSum(), lambda _, n_tokens=n_tokens: n_tokens

        case "randomsum":
            return RandomSum(), lambda ref: len(french_sentencizer(ref))

        case "textrank":
            return (TextRank(sentence_encoder,
                             sentence_pred=lambda s: len(s.split()) > 4,
                             post_process=lambda summary: summary.replace("'", "’")),
                    lambda ref: len(french_sentencizer(ref)))

    raise NotImplementedError(f"Model {model_name} is not implemented.")
