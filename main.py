from pathlib import Path
import re

from tqdm import tqdm
import deal

from LDS.book_loader import BookLoader
from LDS.summarizer_ios import read_references
from LDS.summarizer_ios import output_summaries, output_scores
from LDS.evaluate import evaluate
from LDS.summarizer_factory import summarizer_factory, ModelName
from LDS.nlp_utils import RE_ALPHA
from LDS.textrank import TextRank

# deal.enable()
deal.disable()

MODEL_NAME: ModelName = "textrank"
# NOTE: textrank_french_semantic has the overall best scores
SENTENCE_ENCODER: TextRank.SentenceEncoder = "french_semantic"
SUMMARIES_OUT_PATH = Path("data/output_summaries/").resolve()
SCORES_OUT_PATH = Path("scores/textrank/").resolve()

book = BookLoader(
    doc_path='data/D5627-Dolan.docx',
    markers={
        "slice": [r"^Introduction$", r"Annexe /$"],
        "chapter": r"^Chapitre \d+ /$|^Conclusion$",
        "headers": r"^Chapitre \d+ /.+"
                   r"|^Introduction$"
                   r"|^Stress, santé et performance au travail$"
                   r"|^Conclusion$",
        "footnotes": re.compile(
            r""".+?[A-Z]\.              # At least one character + a capital letter + a dot
                \s.*?                   # + Whitespace + any # of characters
                \(\d{4}\)               # + 4 digits within parens
            """, re.VERBOSE),           # e.g. "12	Zuckerman, M. (1971). Dimensions of ..."
        "undesirables": re.compile(
            r"""^CONFUCIUS$
                |^Matière à réFlexion$
                |^/\tPost-scriptum$
                |^<www\.pbs\.org/bodyandsoul/218/meditation\.htm>.+?\.$
                |^Source\s:\s
            """, re.VERBOSE),
        "citing": re.compile(
            rf"""((?:{RE_ALPHA}){3,}?)  # Capture at least 3 alphabetic characters
                 \d+                    # + at least one digit
            """, re.VERBOSE),           # e.g. "cited1"
        "na_span": [
            # Starts with this:
            r"^exerCiCe \d\.\d /$",
            # Ends with any of these:
            r"^Chapitre \d+ /$"
            r"|^Conclusion$"
            r"|^Les caractéristiques personnelles\."
            r"|/\tLocus de contrôle$"
            r"|^L'observation de sujets a amené Rotter"
            r"|^Lorsqu'une personne souffre de stress"]
    }
)

observed_lengths = [len(c) for c in book.chapters]
expected_lengths = [30155, 48537, 70349, 71779, 87327, 96484, 11090]
assert observed_lengths == expected_lengths
chapters_to_summarize = book.get_chapters(1, 3)
references = read_references(Path("data/references/").resolve())
assert len(chapters_to_summarize) == len(references)

print("GENERATING SUMMARIES PER CHAPTER...")
summarizer, get_summary_len = summarizer_factory( # pylint: disable=unpacking-non-sequence
    MODEL_NAME, sentence_encoder=SENTENCE_ENCODER
)
summary_units = [
    {
        "CHAPTER": idx + 1,
        "SUMMARY": summarizer(chapter, get_summary_len(ref)),
        "REFERENCE": ref
    }
    for idx, (chapter, ref) in
    tqdm(enumerate(zip(chapters_to_summarize, references)),
         total=len(references))
]

model_name = (f"{MODEL_NAME}_{SENTENCE_ENCODER}"
              if MODEL_NAME == "textrank"
              else MODEL_NAME)

output_summaries(
    summary_units,
    out_path=SUMMARIES_OUT_PATH,
    model_name=model_name,
    post_read_sample=False
)

summaries   = [su["SUMMARY"]   for su in summary_units]
references  = [su["REFERENCE"] for su in summary_units]
scores      = evaluate(summaries, references)

output_scores(
    scores,
    out_path=SCORES_OUT_PATH,
    model_name=model_name
)
