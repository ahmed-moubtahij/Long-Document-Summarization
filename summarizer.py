from __future__ import annotations
from pathlib import Path
import json
from typing import ClassVar, Literal

import deal
import jsonlines as jsonl
from more_itertools import chunked_even
import torch
from transformers import EncoderDecoderModel
from transformers import RobertaTokenizerFast
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import SummarizationPipeline
import spacy
from spacy.lang.fr import French

from french_textrank import FrenchTextRank
from book_loader import BookLoader

spacy.prefer_gpu() # type: ignore

@deal.raises(NotImplementedError, ValueError)
@deal.has('io', 'read', 'stdout', 'write')
def main():

    print(f"\nIS CUDA AVAILABLE: {torch.cuda.is_available()}\n")

    MODEL_NAME = "textrank"

    summarizer = make_summarizer(MODEL_NAME, sentence_encoder="french_semantic")
    references = read_references(Path("data/references"))

    chapters_to_summarize = read_chapters(1, 3)

    print("GENERATING SUMMARIES PER CHAPTER...")
    summary_units = [
        {
            "CHAPTER": idx + 1,
            # pylint: disable=not-callable
            "SUMMARY": (summarizer(chapter) # pyright: reportGeneralTypeIssues=false
                        if MODEL_NAME != "textrank"
                        else summarizer(chapter, # pyright: reportGeneralTypeIssues=false
                                        n_sentences=len(FrenchSummarizer.sentencizer(ref)),
                                        sent_pred=lambda s: len(s.split()) > 4)),
            "REFERENCE": ref
        }
        for idx, (chapter, ref) in enumerate(zip(chapters_to_summarize, references))
    ]

    out_path = output_summaries(
        summary_units,
        Path("data/output_summaries/"),
        MODEL_NAME)

    print_sample(out_path)

@deal.has()
@deal.raises(NotImplementedError)
def make_summarizer(
    model_name: Literal["camembert", "mbart", "textrank"],
    sentence_encoder: FrenchTextRank.SentenceEncoder | None = None
) -> FrenchSummarizer | FrenchTextRank:

    match model_name:
        case "camembert":
            return Camembert()

        case "mbart":
            return Mbart()

        case "textrank":
            return FrenchTextRank(sentence_encoder=sentence_encoder)

    raise NotImplementedError(f"model {model_name} is not implemented.")

# TODO: Add a RandomSum then update report with its scores
# TODO: Add https://huggingface.co/plguillou/t5-base-fr-sum-cnndm
class FrenchSummarizer():

    device: ClassVar = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp: ClassVar = French()
    nlp.add_pipe("sentencizer")

    @deal.pure
    @staticmethod
    def trim(text: str) -> str:
        """Removes last sentence. Useful when the decoder generates it incompletely"""
        all_sents_but_last = FrenchSummarizer.sentencizer(text)[:-1]
        return '\n'.join(all_sents_but_last)

    @deal.pure
    @staticmethod
    def sentencizer(text: str) -> list[str]:
        return list(map(str, FrenchSummarizer.nlp(text).sents))

class Mbart(FrenchSummarizer):

    @deal.pure
    def __init__(self) -> None:

        ckpt = 'lincoln/mbart-mlsum-automatic-summarization'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # type: ignore
        self.mbart_nlp = SummarizationPipeline(
            self.model, self.tokenizer,
            device=self.model.device)

    @deal.pure
    def __call__(self, text: str, trim=True) -> str:

        memory_safe_n_chunks = 512

        text_chunks = map(' '.join,
                          chunked_even(text.split(), memory_safe_n_chunks))
        summary = ' '.join(
            self.mbart_nlp(text_chunk,
                           clean_up_tokenization_spaces=True)[0]["summary_text"]
            for text_chunk in text_chunks)

        if trim:
            return self.trim(summary)

        return summary

class Camembert(FrenchSummarizer):

    @deal.safe
    def __init__(self) -> None:

        ckpt = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
        self.model = EncoderDecoderModel.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # type: ignore

    @deal.pure
    def __call__(self, text: str, trim=True) -> str:

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

        if trim:
            return self.trim(summary)

        return summary


@deal.pure
@deal.pre(lambda refs_path: refs_path.exists())
def read_references(refs_path: Path) -> list[str]:
    return [
        ref_file.read_text(encoding="utf-8")
        for ref_file in sorted(refs_path.iterdir())
    ]

@deal.has('read')
@deal.safe
def read_chapters(first_chapter=0, last_chapter: int | None=None) -> list[str]:

    with open('parameters.json', 'r', encoding="utf-8") as json_file:
        params = json.load(json_file)

    book = BookLoader(**params)

    chapters = book.chapters[first_chapter:
                             None if last_chapter is None
                             else last_chapter + 1]

    return ['\n'.join(paragraph for paragraph in chapter[1: ])
            for chapter in chapters]

@deal.has('stdout', 'write')
@deal.safe
@deal.pre(lambda _: _.out_path.exists())
def output_summaries(summary_units: list[dict[str, int | str]],
                     out_path: Path,
                     model_name: str) -> Path:

    out_path /= f"{model_name}_summaries.jsonl"
    with open(out_path, 'w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)

    print(f"Output summaries to {out_path}\n")
    return out_path

@deal.has('io', 'stdout')
@deal.raises(ValueError)
@deal.pre(lambda _: _.out_path.exists())
def print_sample(out_path: Path, just_one=True) -> None:

    with jsonl.open(out_path, mode='r') as summarization_units:
        for summary_unit in summarization_units:
            chapter, summary = (summary_unit["CHAPTER"],
                                summary_unit["SUMMARY"])
            print(f"CHAPTER: {chapter}\n")
            double_spacing_summary = summary.replace('\n', "\n\n")
            print(f"SUMMARY:\n{double_spacing_summary}\n")
            print('-' * 100)
            if just_one:
                break

if __name__ == "__main__":
    main()
