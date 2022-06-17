from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from pathlib import Path
from typing import Any, Iterator, List, Union, Dict, Callable
import re
from itertools import chain, groupby, islice
import docx
from simplify_docx import simplify

# TODO: Import `BookLoader` into `data_analysis.ipynb`

def main(args) -> None:
    data_path = make_data_path(args.data_dir, args.data_fn)
    book = BookLoader(data_path, tables_of_contents_idx=152)
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

class BookLoader:

    def __init__(self, data_path: Path,
                 re_separator=r"^Chapitre (\d+) \/$",
                 tables_of_contents_idx=0):
        assert data_path.exists()
        self.docx: dict = simplify(docx.Document(data_path))
        self.tables_of_contents_idx = tables_of_contents_idx
        self.paragraphs = self._init_paragraphs()
        self.re_separator = re_separator
        self.chapters = self._init_chapters()

    # TODO: Look into aliasing `[Union[str, List[dict]]]` as text_or_tables
    def _chapter_indexer(self) -> Callable[[Union[str, List[dict]]], int]:
        separator_pattern = re.compile(self.re_separator)
        current_chapter = 0

        def indexer(paragraph: Union[str, List[dict]]) -> int:
            nonlocal current_chapter
            if isinstance(paragraph, str):
                if found_chapter := separator_pattern.search(paragraph):
                    current_chapter = int(found_chapter.group(1))
            return current_chapter

        return indexer

    # TODO: Look into aliasing `[Union[str, List[dict]]]` as text_or_tables
    def _strip_headers(self, paragraphs: Iterator[Union[str, List[dict]]]) -> List[Union[str, List[dict]]]:
        page_headers = ["Stress, santé et performance au travail", "Introduction"]
        header_pattern = re.sub(r"\(|\)", '', self.re_separator).removesuffix('$') + r".*"
        return [p for p in paragraphs
                if isinstance(p, str)
                and p not in page_headers
                and not re.match(header_pattern, p)]

    def _init_chapters(self) -> Dict[int, List[Union[str, List[dict]]]]:
        _chapters = groupby(self.paragraphs, self._chapter_indexer())
        chapters = {idx: self._strip_headers(paragraphs) for idx, paragraphs in _chapters}
        return chapters

    def _init_paragraphs(self) -> Iterator[Union[str, List[dict]]]:
        _docx: List[dict] = self.docx["VALUE"][0]["VALUE"]

        _paragraphs: Iterator[Any]
        _paragraphs = map(lambda p: p["VALUE"], _docx)
        _paragraphs = filter(lambda p: p != "[w:sdt]", _paragraphs)
        _paragraphs = chain.from_iterable(_paragraphs)
        _paragraphs = filter(lambda p: p["TYPE"] != "CT_Empty", _paragraphs)
        _paragraphs = map(lambda p: p["VALUE"], _paragraphs)
        _paragraphs = islice(_paragraphs, self.tables_of_contents_idx, None)

        return _paragraphs


if __name__ == "__main__":
    main(get_args())
