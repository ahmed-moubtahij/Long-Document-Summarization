# from __future__ import annotations
from abc import ABC, abstractmethod
from functools import partial
from typing import ClassVar

import deal
from funcy_chain import IterChain
import torch
from transformers import EncoderDecoderModel
from transformers import RobertaTokenizerFast
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import SummarizationPipeline

from LDS.nlp_utils import trim

class AbstractiveSummarizer(ABC):
    device: ClassVar = 'cuda'

    @deal.pre(lambda _: torch.cuda.is_available)
    @deal.has('read', 'network')
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def __call__(self, text: str, len_in_tokens: int) -> str:
        ...

class MbartSum(AbstractiveSummarizer):

    @deal.raises(ImportError, ValueError, TypeError)
    def __init__(self) -> None:

        super().__init__()
        ckpt = 'lincoln/mbart-mlsum-automatic-summarization'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        self.model = self.model.to(self.device)

        mbart_nlp = SummarizationPipeline(
            self.model,
            self.tokenizer,
            device=self.model.device
        )
        self.mbart_summarizer = partial(
            mbart_nlp,
            clean_up_tokenization_spaces=True
        )

    @deal.safe
    @deal.has('stderr')
    def __call__(self, text: str, len_in_tokens: int) -> str:

        return (IterChain(text.split())
                    .chunks(len_in_tokens)
                    .map(' '.join)
                    .map(self.mbart_summarizer)
                    .map(lambda o: o[0]["summary_text"])
                    .thru(' '.join)
                    .thru(trim)
               ).value

class CamembertSum(AbstractiveSummarizer):

    @deal.safe
    def __init__(self) -> None:
        super().__init__()
        ckpt = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
        self.model = EncoderDecoderModel.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # pyright: ignore[reportGeneralTypeIssues]

    def __call__(self, text: str, len_in_tokens: int) -> str:

        inputs = self.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # https://huggingface.co/docs/transformers/main_classes/text_generation
        output = self.model.generate( # pyright: ignore[reportGeneralTypeIssues]
            input_ids,
            attention_mask=attention_mask,
            min_length=len_in_tokens,
            max_length=len_in_tokens,
            repetition_penalty=0.65,
            num_beams=10
        )

        summary = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return trim(summary)
