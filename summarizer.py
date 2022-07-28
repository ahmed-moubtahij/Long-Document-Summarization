from pathlib import Path
import json
from typing import Literal
import jsonlines as jsonl
from more_itertools import chunked_even
import torch
from tqdm import tqdm
from transformers import EncoderDecoderModel    # type: ignore
from transformers import RobertaTokenizerFast   # type: ignore
from transformers import AutoTokenizer          # type: ignore
from transformers import AutoModelForSeq2SeqLM  # type: ignore
from transformers import SummarizationPipeline  # type: ignore
import spacy
from spacy.lang.fr import French

from french_textrank import FrenchTextRank
from book_loader import BookLoader

spacy.prefer_gpu() # type: ignore

def main():

    print(f"\nIS CUDA AVAILABLE: {torch.cuda.is_available()}\n")

    references_dir = Path("data/references").expanduser().resolve()
    assert references_dir.exists()
    references = [ref_file.read_text(encoding="utf-8")
                  for ref_file in sorted(references_dir.iterdir())]

    MODEL_NAME = "textrank" # pylint: disable=invalid-name

    summarizer = FrenchSummarizer(MODEL_NAME)
    ref_lens = [len(FrenchSummarizer.sentencizer(ref)) for ref in references]
    print("GENERATING SUMMARIES PER CHAPTER...")

    chapters_to_summarize = read_chapters(1, 3)
    summaries = [summarizer(chapter, trim=False, textrank_n_sentences=n_sents)
                 for chapter, n_sents in tqdm(zip(chapters_to_summarize, ref_lens))]

    assert len(summaries) == len(references)

    summary_units = [
        {"CHAPTER": idx + 1, "SUMMARY": summary, "REFERENCE": ref}
        for idx, (summary, ref) in enumerate(zip(summaries, references))
    ]

    out_path = Path("data/output_summaries/").expanduser().resolve()
    assert out_path.exists()
    out_path /= f"{MODEL_NAME}_summaries.jsonl"

    with open(out_path, 'w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)

    with jsonl.open(out_path, mode='r') as summarization_units:
        for summary_unit in summarization_units:
            chapter, summary = (summary_unit["CHAPTER"],
                                summary_unit["SUMMARY"])
            print(f"CHAPTER: {chapter}\n")
            print(f"SUMMARY:\n{summary}\n")
            print('-' * 100)

class FrenchSummarizer():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp = French()
    nlp.add_pipe("sentencizer")

    ModelName = Literal["camembert", "barthez", "textrank"]

    def __init__(self, model_name: ModelName) -> None:

        match model_name:
            case "camembert":
                print("model_name: camembert\n")
                ckpt = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
                self.tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
                self.model = EncoderDecoderModel.from_pretrained(ckpt)
                self.model = self.model.to(self.device) # type: ignore
            case "barthez":
                print("model_name: barthez\n")
                ckpt = "moussaKam/barthez-orangesum-abstract"
                self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
                self.model = self.model.to(self.device) # type: ignore
            case "mbart":
                print("model_name: mbart\n")
                ckpt = 'lincoln/mbart-mlsum-automatic-summarization'
                self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
                self.model = self.model.to(self.device) # type: ignore
                self.mbart_nlp = SummarizationPipeline(
                    self.model, self.tokenizer,
                    device=self.model.device)
            case "textrank":
                print("model_name: textrank\n")
                self.french_text_rank = FrenchTextRank()

        self.model_name = model_name

    def __call__(self, text: str, trim=False, textrank_n_sentences: int | None=None) -> str:

        if self.model_name == "textrank":
            return self.french_text_rank.summarize(
                text,
                textrank_n_sentences, # type: ignore
                sent_pred=lambda sent: len(sent.split()) > 4)

        if self.model_name == "mbart":
            memory_safe_n_chunks = 512
            text_chunks = map(' '.join,
                              chunked_even(text.split(), memory_safe_n_chunks))
            summary = ' '.join(
                self.mbart_nlp(text_chunk,
                               clean_up_tokenization_spaces=True)[0]["summary_text"]
                for text_chunk in text_chunks)

            return summary

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

    @staticmethod
    def trim(text: str) -> str:
        """Removes last sentence. Useful when the decoder generates it incompletely"""
        all_sents_but_last = FrenchSummarizer.sentencizer(text)[:-1]

        return '\n'.join(all_sents_but_last)

    @staticmethod
    def sentencizer(text: str) -> list[str]:

        return list(map(str, FrenchSummarizer.nlp(text).sents))

def read_chapters(first_chapter=0, last_chapter: int | None=None) -> list[str]:

    with open('parameters.json', 'r', encoding="utf-8") as json_file:
        params = json.load(json_file)

    book = BookLoader(**params)

    chapters = book.chapters[first_chapter:
                             None if last_chapter is None
                             else last_chapter + 1]

    return ['\n'.join(paragraph for paragraph in chapter[1: ])
            for chapter in chapters]

if __name__ == "__main__":
    main()