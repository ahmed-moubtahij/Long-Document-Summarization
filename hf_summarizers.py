from __future__ import annotations
from pathlib import Path
from typing import ClassVar, Literal

import deal
from more_itertools import chunked_even
import torch
from transformers import EncoderDecoderModel
from transformers import RobertaTokenizerFast
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import SummarizationPipeline

from book_loader import read_chapters
from summarizer_ios import read_references
from summarizer_ios import output_summaries
from summarizer_ios import print_sample
from nlp_utils import trim

@deal.raises(NotImplementedError, ValueError, TypeError)
@deal.has('io')
def main():

    print(f"\nIS CUDA AVAILABLE: {torch.cuda.is_available()}\n")

    MODEL_NAME = "camembert"

    summarizer = summarizer_factory(MODEL_NAME)
    references = read_references(Path("data/references"))

    chapters_to_summarize = read_chapters(1, 3)

    print("GENERATING SUMMARIES PER CHAPTER...")
    summarizer: FrenchSummarizer # For some reason pylint needs this
    summary_units = [
        {
            "CHAPTER": idx + 1,
            "SUMMARY": summarizer(chapter),
            "REFERENCE": ref
        }
        for idx, (chapter, ref) in enumerate(zip(chapters_to_summarize, references))
    ]

    out_path = output_summaries(
        summary_units, Path("data/output_summaries/"), MODEL_NAME)

    print_sample(out_path)

# TODO: Add a RandomSum then update report with its scores
# TODO: Add https://huggingface.co/plguillou/t5-base-fr-sum-cnndm
@deal.raises(NotImplementedError)
@deal.has('read', 'network', 'stderr')
def summarizer_factory(
    model_name: Literal["camembert", "mbart"]
) -> FrenchSummarizer:

    match model_name:
        case "camembert":
            return CamembertSum()

        case "mbart":
            return MbartSum()

    raise NotImplementedError(f"model {model_name} is not implemented.")

model_init_contract = deal.chain(
    deal.has('read', 'network'),
    deal.safe
)

class FrenchSummarizer():
    device: ClassVar = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, text: str, trim_last_sent=True):
        pass

class MbartSum(FrenchSummarizer):

    @model_init_contract
    def __init__(self) -> None:

        ckpt = 'lincoln/mbart-mlsum-automatic-summarization'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # type: ignore
        self.mbart_nlp = SummarizationPipeline(
            self.model, self.tokenizer,
            device=self.model.device)

    @deal.safe
    @deal.has('stderr')
    def __call__(self, text: str, trim_last_sent=True) -> str:

        memory_safe_n_chunks = 512

        text_chunks = map(' '.join,
                          chunked_even(text.split(), memory_safe_n_chunks))
        summary = ' '.join(
            self.mbart_nlp(text_chunk,
                           clean_up_tokenization_spaces=True)[0]["summary_text"]
            for text_chunk in text_chunks)

        if trim_last_sent:
            return trim(summary)

        return summary

class CamembertSum(FrenchSummarizer):

    @model_init_contract
    def __init__(self) -> None:

        ckpt = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
        self.model = EncoderDecoderModel.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # type: ignore

    @deal.pure
    def __call__(self, text: str, trim_last_sent=True) -> str:

        inputs = self.tokenizer(
            [text], padding="max_length",
            truncation=True, max_length=512,
            return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        min_length = self.model.config.max_length * 8 # type: ignore
        max_length = min_length
        # https://huggingface.co/docs/transformers/main_classes/text_generation
        output = self.model.generate( # type: ignore
            input_ids,
            attention_mask=attention_mask,
            min_length=min_length,
            max_length=max_length,
            repetition_penalty=0.65,
            num_beams=10)

        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if trim_last_sent:
            return trim(summary)

        return summary

if __name__ == "__main__":
    main()
