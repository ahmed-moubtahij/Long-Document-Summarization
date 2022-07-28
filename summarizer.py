from pathlib import Path
from collections.abc import Callable
import json
import jsonlines as jsonl
import torch
from tqdm import tqdm
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
import spacy
from spacy.lang.fr import French
from book_loader import BookLoader
spacy.prefer_gpu() # type: ignore

def main() -> None:

    print(f"\nIS CUDA AVAILABLE: {torch.cuda.is_available()}\n")

    summaries, references = get_summaries_and_refs()

    summary_units = [
        {"chapter": idx + 1, "summary": summary, "reference": ref}
        for idx, (summary, ref) in enumerate(zip(summaries, references))
    ]

    out_path = Path("data/").expanduser().resolve()
    assert out_path.exists()
    out_path /= "summaries.jsonl"

    with open(out_path, 'w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)

    with jsonl.open(out_path) as summarization_units:
        for summary_unit in summarization_units:
            chapter, summary, reference = (summary_unit["chapter"],
                                           summary_unit["summary"],
                                           summary_unit["reference"])
            print(f"CHAPTER: {chapter}\n")
            print(f"SUMMARY: {summary}\n")
            print(f"REFERENCE: {reference}")
            print('-' * 40)

def get_summaries_and_refs() -> tuple[list[str], list[str]]:

    references_dir = Path("data/references").expanduser().resolve()
    assert references_dir.exists()
    references = [ref_file.read_text(encoding="utf-8")
                  for ref_file in sorted(references_dir.iterdir())]

    chapters_to_summarize = ['\n'.join(p for p in chapter)
                             for chapter in read_chapters(1, 3)]

    print("GENERATING SUMMARIES PER CHAPTER...")
    summaries = [trim(generate_summary(chapter))
                 for chapter in tqdm(chapters_to_summarize)]

    assert len(summaries) == len(references)

    return summaries, references

def read_chapters(first_chapter=0, last_chapter: int | None=None) -> list[list[str]]:

    with open('parameters.json', 'r', encoding="utf-8") as json_file:
        params = json.load(json_file)

    book = BookLoader(**params)

    observed_lengths = [len(c) for c in book.chapters]
    expected_lengths = [44, 136, 194, 178, 345, 348, 29]
    assert observed_lengths == expected_lengths

    if last_chapter is None:
        return book.chapters[first_chapter: ]

    return book.chapters[first_chapter: last_chapter + 1]

def generate_summary(text: str) -> str:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization'
    tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt).to(device) # type: ignore

    inputs = tokenizer([text], padding="max_length",
                       truncation=True, max_length=512,
                       return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    output = model.generate(input_ids,
                            attention_mask=attention_mask,
                            min_length=model.config.max_length * 8,
                            max_length=model.config.max_length * 8,
                            repetition_penalty=0.6,
                            num_beams=10)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def french_sentencizer() -> Callable[[str], list[str]]:
    nlp = French()
    nlp.add_pipe("sentencizer")

    def _sentencize(text):
        sents = map(str, nlp(text).sents)
        return list(sents)

    return _sentencize

# Remove last sentence as the decoder tends to generate it incompletely
def trim(text: str) -> str:
    all_sents_but_last = french_sentencizer()(text)[:-1]
    return '\n'.join(all_sents_but_last)

if __name__ == "__main__":
    main()
