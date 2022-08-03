from __future__ import annotations
from pathlib import Path
import json
from typing import ClassVar, Literal
import deal
import funcy as fy
from funcy_chain import IterChain
import jsonlines as jsonl
# from more_itertools import chunked_even
import torch
from tqdm import tqdm
# pyright: reportPrivateImportUsage=false
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

def main():

    print(f"\nIS CUDA AVAILABLE: {torch.cuda.is_available()}\n")

    MODEL_NAME = "textrank" # pylint: disable=invalid-name

    summarizer = make_summarizer(MODEL_NAME)
    references = read_references(Path("data/references"))

    ref_lens = [len(FrenchSummarizer.sentencizer(ref)) for ref in references]

    print("GENERATING SUMMARIES PER CHAPTER...")
    summaries = [
        summarizer(chapter, # pylint: disable=not-callable
                   n_sentences=ref_n_sents,
                   sent_pred=lambda s: len(s.split()) > 4)
        for chapter, ref_n_sents in tqdm(zip(read_chapters(1, 3), ref_lens))
    ]
    assert len(summaries) == len(references) # postcond

    summary_units = [
        {"CHAPTER": idx + 1, "SUMMARY": summary, "REFERENCE": ref}
        for idx, (summary, ref) in enumerate(zip(summaries, references))
    ]

    # TODO: Regen all summaries for all models
    out_path = output_summaries(
        summary_units,
        Path("data/output_summaries/"),
        MODEL_NAME)

    with jsonl.open(out_path, mode='r') as summarization_units:
        for summary_unit in summarization_units:
            chapter, summary = (summary_unit["CHAPTER"],
                                summary_unit["SUMMARY"])
            print(f"CHAPTER: {chapter}\n")
            print(f"SUMMARY:\n{summary}\n")
            print('-' * 100)

def make_summarizer(
    model_name: Literal["camembert", "barthez", "mbart", "textrank"]
) -> FrenchSummarizer | FrenchTextRank:

    match model_name:
        case "camembert":
            return Camembert()

        case "barthez":
            return Barthez()

        case "mbart":
            return Mbart()

        case "textrank":
            return FrenchTextRank()

    raise NotImplementedError(f"model {model_name} is not implemented.")

class FrenchSummarizer():

    device: ClassVar = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp: ClassVar = French()
    nlp.add_pipe("sentencizer")

    @staticmethod
    def trim(text: str) -> str:
        """Removes last sentence. Useful when the decoder generates it incompletely"""
        all_sents_but_last = FrenchSummarizer.sentencizer(text)[:-1]
        return '\n'.join(all_sents_but_last)

    @staticmethod
    def sentencizer(text: str) -> list[str]:
        return list(map(str, FrenchSummarizer.nlp(text).sents))

class Mbart(FrenchSummarizer):

    def __init__(self) -> None:

        ckpt = 'lincoln/mbart-mlsum-automatic-summarization'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # type: ignore
        self.mbart_nlp = SummarizationPipeline(
            self.model, self.tokenizer,
            device=self.model.device)

    def __call__(self, text: str, trim=False) -> str:

        memory_safe_n_chunks = 512
        summarize_chunk = fy.rcompose(
            fy.partial(self.mbart_nlp, clean_up_tokenization_spaces=True),
            fy.partial(fy.get_in, path=[0, "summary_text"]))

        summary = ' '.join(
            IterChain(text.split())
            .chunks(memory_safe_n_chunks)
            .thru(' '.join)
            .thru(summarize_chunk)
            .value)

        # text_chunks = map(' '.join,
        #                   chunked_even(text.split(), memory_safe_n_chunks))
        # summary = ' '.join(
        #     self.mbart_nlp(text_chunk,
        #                    clean_up_tokenization_spaces=True)[0]["summary_text"]
        #     for text_chunk in text_chunks)

        if trim:
            return self.trim(summary)

        return summary

class Barthez(FrenchSummarizer):

    def __init__(self) -> None:

        ckpt = "moussaKam/barthez-orangesum-abstract"
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # type: ignore

    def __call__(self, text: str, trim: bool = False) -> str:

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

class Camembert(FrenchSummarizer):

    def __init__(self) -> None:

        ckpt = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
        self.model = EncoderDecoderModel.from_pretrained(ckpt)
        self.model = self.model.to(self.device) # type: ignore

    def __call__(self, text: str, trim: bool = False) -> str:

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


@deal.pre(lambda refs_path: refs_path.exists())
def read_references(refs_path: Path) -> list[str]:
    return [
        ref_file.read_text(encoding="utf-8")
        for ref_file in sorted(refs_path.iterdir())
    ]

def read_chapters(first_chapter=0, last_chapter: int | None=None) -> list[str]:

    with open('parameters.json', 'r', encoding="utf-8") as json_file:
        params = json.load(json_file)

    book = BookLoader(**params)

    chapters = book.chapters[first_chapter:
                             None if last_chapter is None
                             else last_chapter + 1]

    return ['\n'.join(paragraph for paragraph in chapter[1: ])
            for chapter in chapters]

@deal.pre(lambda out_path: out_path.exists())
def output_summaries(summary_units: list[dict[str, int | str]],
                     out_path: Path,
                     model_name: str) -> Path:

    out_path /= f"{model_name}_summaries.jsonl"
    with open(out_path, 'w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)

    return out_path

if __name__ == "__main__":
    main()
