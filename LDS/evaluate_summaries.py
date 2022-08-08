"""
In case of PythonRouge errors:
* UnicodeEncodeError: 'ascii' codec can't encode character '\xe9'
    - add argument `encoding='utf-8'` to all file open's in pythonrouge.py
* subprocess.CalledProcessError:
    -sudo apt-get install libxml-parser-perl
"""
from pathlib import Path
import json
from typing import TypeAlias

import jsonlines as jsonl
import deal
from pythonrouge.pythonrouge import Pythonrouge # type: ignore

from nlp_utils import french_sentencizer

# TODO: typehint and contract this

@deal.has('io', 'stdout', 'write')
@deal.raises(AssertionError, ValueError)
def main():

    # TODO: `EXPERIMENT_ID` should be synchronized with summarizer
    MODEL_NAME = "textrank"
    EXPERIMENT_ID = f"{MODEL_NAME}_flaubert_base_uncased-xnli-sts"
    summaries_fp = f"data/output_summaries/{EXPERIMENT_ID}_summaries.jsonl"
    output_scores(summaries_fp, EXPERIMENT_ID)

@deal.has('io', 'stdout', 'write')
@deal.raises(AssertionError, ValueError)
def output_scores(summaries_fp: str, model_name: str):

    _summaries_fp = Path(summaries_fp).expanduser().resolve()
    assert _summaries_fp.exists(), f"Unable to find {_summaries_fp}\n"

    scores_path = Path("scores/").expanduser().resolve()
    assert scores_path.exists()
    scores_path /= f"{model_name}_scores.json"

    with jsonl.open(summaries_fp, mode='r') as summarization_units:
        summaries, references = rouge_preproc(summarization_units)

    score: dict[str, float] = calc_rouge_score(summaries, references)
    print(score)

    with open(scores_path, 'w', encoding="utf-8") as score_writer:
        json.dump(score, score_writer)
    print(f"Scores written to:\n{scores_path}\n")


@deal.pure
def calc_rouge_score(summaries, references) -> dict[str, float]:

    python_rouge = Pythonrouge(
        summary_file_exist=False,
        summary=summaries, reference=references,
        recall_only=False, f_measure_only=False,
        n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
        stemming=True, stopwords=False,
        word_level=True, length_limit=False,
        scoring_formula="average", resampling=True, samples=1000
    )

    return python_rouge.calc_score()


PythonRougeSummaries: TypeAlias = list[list[str]]
PythonRougeReferences: TypeAlias = list[list[list[str]]]
@deal.pure
def rouge_preproc(
    summarization_units: jsonl.Reader
) -> tuple[PythonRougeSummaries, PythonRougeReferences]:
    # Each summarization unit has a system summary
    # A system summary must be a list of sentences
    all_summaries = []

    # Each summarization unit has one or more references
    # Each reference must be a list of sentences
    all_references = []

    # TODO: Make sure the contents are correct and correctly aligned
    #       and remove '\n's
    for summ_unit in summarization_units:
        summary_sents = french_sentencizer(summ_unit["SUMMARY"])
        all_summaries.append(summary_sents)

        reference_sents = french_sentencizer(summ_unit["REFERENCE"])
        all_references.append([reference_sents])

    return all_summaries, all_references

if __name__ == "__main__":
    main()
