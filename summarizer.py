from pathlib import Path
import re
from collections.abc import Callable
import jsonlines as jsonl
import torch
from tqdm import tqdm
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from spacy.lang.fr import French
from book_loader import BookLoader

def generate_summary(text: str) -> str:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUSING {device.upper()}\n")
    ckpt = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization'
    tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

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



# Remove last sentence as the decoder tends to generate it incompletely
def trim() -> Callable[[str], str]:
    nlp = French()
    nlp.add_pipe("sentencizer")

    def trimmer(text: str):
        sents = map(str, nlp(text).sents)
        all_sents_but_last = list(sents)[:-1]
        return '\n'.join(all_sents_but_last)

    return trimmer

def main() -> None:

    print(f"\nIS CUDA AVAILABLE: {torch.cuda.is_available()}\n")

    doc_path = Path("data/D5627-Dolan.docx").expanduser().resolve()

    start_marker = r"^Introduction$"
    slice_markers = (start_marker, re.compile(r"^Annexe /$"))
    conclusion_marker = r"^Conclusion$"
    compiled_header_marker = re.compile(
        rf"^Chapitre \d+ /.+"
        rf"|{start_marker}"
        rf"|^Stress, santé et performance au travail$"
        rf"|{conclusion_marker}")
    chapter_marker = rf"^Chapitre \d+ /$|{conclusion_marker}"
    na_span_markers = (
            r"^exerCiCe \d\.\d /$",
            '|'.join([chapter_marker,
                      r"^Les caractéristiques personnelles\.",
                      r"/\tLocus de contrôle$",
                      r"^L'observation de sujets a amené Rotter",
                      r"^Lorsqu'une personne souffre de stress"]))

    book = BookLoader(doc_path,
                      {"slice_markers": slice_markers,
                       "chapter_marker": chapter_marker,
                       "header_marker": compiled_header_marker,
                       "na_span_markers": na_span_markers})

    expected_lengths = [44, 136, 194, 178, 345, 348, 29]
    assert [len(c) for c in book.chapters] == expected_lengths

    references_dir = Path("data/references").expanduser().resolve()
    assert references_dir.exists()
    references = [ref_file.read_text(encoding="utf-8")
                  for ref_file in sorted(references_dir.iterdir())]

    chapters_to_summarize = ['\n'.join(p for p in chapter)
                             for chapter in book.chapters[1: -3]]

    summaries = [trim()(generate_summary(chapter))
                 for chapter in tqdm(chapters_to_summarize)]

    assert len(summaries) == len(references)

    summary_units = [
        {"chapter": idx + 1, "summary": summary, "reference": ref}
        for idx, (summary, ref) in enumerate(zip(summaries, references))
    ]

    out_path = Path("data/").expanduser().resolve()
    assert out_path.exists()

    with open(out_path/"summaries.jsonl", 'w', encoding='utf-8') as out_jsonl:
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

if __name__ == "__main__":
    main()
