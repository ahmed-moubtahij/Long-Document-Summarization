from pathlib import Path
import json
import jsonlines as jsonl
from more_itertools import chunked_even
import torch
from tqdm import tqdm
# pylint: disable=unused-import
# from transformers import EncoderDecoderModel, RobertaTokenizerFast # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # type: ignore
from transformers import SummarizationPipeline
import spacy
from spacy.lang.fr import French
from book_loader import BookLoader
spacy.prefer_gpu() # type: ignore

def main() -> None:

    print(f"\nIS CUDA AVAILABLE: {torch.cuda.is_available()}\n")

    # model_name = "camembert"
    # ckpt = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization'
    # tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
    # model = EncoderDecoderModel.from_pretrained(ckpt)

    # model_name = "barthez"
    # tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-abstract")
    # model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-abstract")
    # Result: Barthez is trained on *title*-summarization -> short summaries.

    model_name = "mbart"
    ckpt = 'lincoln/mbart-mlsum-automatic-summarization'
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)



    summarizer = Summarizer(tokenizer, model)
    print("GENERATING SUMMARIES PER CHAPTER...")
    summaries = [summarizer(chapter, trim=True)
                 for chapter in tqdm(read_chapters(1, 3))]

    references_dir = Path("data/references").expanduser().resolve()
    assert references_dir.exists()
    references = [ref_file.read_text(encoding="utf-8")
                  for ref_file in sorted(references_dir.iterdir())]

    assert len(summaries) == len(references)

    summary_units = [
        {"chapter": idx + 1, "summary": summary, "reference": ref}
        for idx, (summary, ref) in enumerate(zip(summaries, references))
    ]

    out_path = Path("data/output_summaries/").expanduser().resolve()
    assert out_path.exists()
    out_path /= f"{model_name}_summaries.jsonl"

    with open(out_path, 'w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)

    with jsonl.open(out_path, mode='r') as summarization_units:
        for summary_unit in summarization_units:
            chapter, summary = (summary_unit["chapter"],
                                summary_unit["summary"])
            print(f"CHAPTER: {chapter}\n")
            print(f"SUMMARY: {summary}\n")
            print('-' * 100)

class Summarizer():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp = French()
    nlp.add_pipe("sentencizer")

    def __init__(self, tokenizer, model) -> None:

        self.tokenizer = tokenizer
        self.model = model.to(self.device)

        self.mbart_nlp = None
        if "lincoln/mbart" in self.model.name_or_path:
            self.mbart_nlp = SummarizationPipeline(
                self.model, self.tokenizer,
                device=self.model.device)

    def __call__(self, text: str, trim=False) -> str:

        if self.mbart_nlp is not None:
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

        min_length = self.model.config.max_length * 8
        max_length = min_length
        # https://huggingface.co/docs/transformers/main_classes/text_generation
        output = self.model.generate(
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
        all_sents_but_last = Summarizer.sentencizer(text)[:-1]

        return '\n'.join(all_sents_but_last)

    @staticmethod
    def sentencizer(text:str) -> list[str]:

        return list(map(str, Summarizer.nlp(text).sents))

def read_chapters(first_chapter=0, last_chapter: int | None=None) -> list[str]:

    with open('parameters.json', 'r', encoding="utf-8") as json_file:
        params = json.load(json_file)

    book = BookLoader(**params)

    observed_lengths = [len(c) for c in book.chapters]
    expected_lengths = [44, 136, 194, 178, 345, 348, 29]
    assert observed_lengths == expected_lengths

    chapters = book.chapters[first_chapter:
                             None if last_chapter is None
                             else last_chapter + 1]

    return ['\n'.join(paragraph for paragraph in chapter)
            for chapter in chapters]

if __name__ == "__main__":
    main()
