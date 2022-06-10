from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from pathlib import Path
from typing import Any, Iterator, List, Union, Dict
import re
from itertools import chain, groupby
import docx
from simplify_docx import simplify

# TODO: write the BookLoader class then import it in data_analysis.ipynb

def main(args) -> None:
    data_path = make_data_path(args.data_dir, args.data_fn)

    book = BookLoader(data_path)

    print(book.chapters)

def make_data_path(data_dir: str, data_fn: str) -> Path:
    data_path = Path(data_dir + data_fn).expanduser().resolve()
    assert data_path.exists()
    return data_path


def get_args() -> argparse.Namespace:
    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-d', '--data-dir', type=str,
                            default="data/",
                            help="Path to directory containing the input data file")
    arg_parser.add_argument('-f', "--data-fn", type=str,
                            default="D5627-Dolan.docx",
                            help=("File name with the text to summarize"))
    return arg_parser.parse_args()


class BookLoader: # pylint: disable=too-few-public-methods

    def __init__(self, data_path: Path, intro_idx=152,
                 n_chapters=5, re_separator=r"Chapitre (\d+) /"):
        assert data_path.exists()
        self.docx: dict = simplify(docx.Document(data_path))
        self.paragraphs: Iterator[Union[str, List[dict]]] = self._init_paragraphs()

        self.intro_idx = intro_idx
        self.n_chapters = n_chapters
        self.re_separator = re_separator
        self.chapters = self._init_chapters()

    class ChapterIndexer: # pylint: disable=too-few-public-methods
        def __init__(self, re_separator: str):
            self.separator_pattern: re.Pattern = re.compile(re_separator)
            self.current_chapter = 0

        def __call__(self, paragraph: Union[str, List[dict]]):
            if isinstance(paragraph, str):
                if found_chapter := self.separator_pattern.search(paragraph):
                    self.current_chapter = found_chapter.group(1)

            return self.current_chapter


    def _init_chapters(self) -> Dict[int, List[Union[str, List[dict]]]]:
        _chapters = groupby(self.paragraphs,
                            self.ChapterIndexer(self.re_separator))
        chapters = {int(idx): list(paragraphs) for idx, paragraphs in _chapters}
        # TODO: Look into an "aggregate_chapter(paragraphs) -> List[str]", but first,
        # look the input format expected for the stats you want
        return chapters

    def _init_paragraphs(self) -> Iterator[Union[str, List[dict]]]:
        _docx: List[dict] = self.docx["VALUE"][0]["VALUE"]
        _paragraphs: Iterator[Any]
        _paragraphs = map(lambda p: p["VALUE"], _docx)
        _paragraphs = filter(lambda p: p != "[w:sdt]", _paragraphs)
        _paragraphs = chain.from_iterable(_paragraphs)
        _paragraphs = filter(lambda p: p["TYPE"] != "CT_Empty", _paragraphs)

        return map(lambda p: p["VALUE"], _paragraphs)


if __name__ == "__main__":
    main(get_args())
