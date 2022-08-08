from pathlib import Path

import deal

from LDS.book_loader import BookLoader
from LDS.hf_summarizers import FrenchSummarizer
from LDS.hf_summarizers import summarizer_factory
from LDS.summarizer_ios import read_references
from LDS.summarizer_ios import output_summaries
from LDS.summarizer_ios import print_sample

@deal.raises(NotImplementedError, ValueError, TypeError)
@deal.has('io')
def main():

    MODEL_NAME = "camembert"

    book = BookLoader.from_params_json()

    chapters_to_summarize = book.get_chapters(1, 3)

    print("GENERATING SUMMARIES PER CHAPTER...")
    summarizer = summarizer_factory(MODEL_NAME)
    summarizer: FrenchSummarizer # For some reason pylint needs this
    references = read_references(Path("data/references"))
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

if __name__ == "__main__":
    main()
