from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from pathlib import Path
from typing import Any, Iterator, List, Union, Dict, Callable
import re
from itertools import chain, dropwhile, groupby
import docx
from simplify_docx import simplify


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

class BookLoader:

    # pylint: disable=E0602 # undefined-variable
    Table = List[dict]
    TextOrTable = Union[str, Table]

    def __init__(self,
                 data_path: Path,
                 title="",
                 intro_marker=r"^Introduction$",
                 chapter_separator=r"^Chapitre (\d+) \/$",
                 header_marker=""):
        assert data_path.exists()
        self.docx: dict = simplify(docx.Document(data_path))

        self.paragraphs = self._init_paragraphs()
        self.title = next(self.paragraphs) if not title else title
        self.intro_marker = intro_marker
        self.chapter_separator = chapter_separator
        self.header_marker = (re.sub(r"\(|\)", '', self.chapter_separator)
                              .removesuffix('$') +  r".*"
                              if not header_marker
                              else header_marker)
        self.chapters = self._init_chapters()

    def _chapter_indexer(self) -> Callable[[TextOrTable], int]:
        separator_pattern = re.compile(self.chapter_separator)
        current_chapter = 0

        def indexer(paragraph: Union[str, List[dict]]) -> int:
            nonlocal current_chapter
            if isinstance(paragraph, str):
                if found_chapter := separator_pattern.search(paragraph):
                    current_chapter = int(found_chapter.group(1))
            return current_chapter

        return indexer

    def _strip_headers(self, paragraphs: Iterator[TextOrTable]) -> Iterator[TextOrTable]:
        return (p for p in paragraphs
                if not isinstance(p, str) # not str => Table => not a header
                or (p != self.title
                    and not re.match(self.intro_marker, p)
                    and not re.match(self.header_marker, p)))


    def _init_chapters(self) -> Dict[int, List[TextOrTable]]:
        paragraphs_from_intro = dropwhile(
            lambda p: (not isinstance(p, str)
                       or not re.match(self.intro_marker, p)),
            self.paragraphs)

        _chapters = groupby(paragraphs_from_intro, self._chapter_indexer())
        chapters = {idx: list(self._strip_headers(paragraphs))
                    for idx, paragraphs in _chapters}
        return chapters

    def _init_paragraphs(self) -> Iterator[TextOrTable]:
        _docx: List[dict] = self.docx["VALUE"][0]["VALUE"]

        _paragraphs: Iterator[Any]
        _paragraphs = map(lambda p: p["VALUE"], _docx)
        _paragraphs = filter(lambda p: p != "[w:sdt]", _paragraphs)
        # https://github.com/microsoft/Simplify-Docx/blob/
        # ce493b60e3e4308bde7399257426c0c68a6c699b/src/simplify_docx/iterators/body.py#L60
        _paragraphs = chain.from_iterable(_paragraphs)
        _paragraphs = filter(lambda p: p["TYPE"] != "CT_Empty", _paragraphs)
        _paragraphs = map(lambda p: p["VALUE"], _paragraphs)
        _paragraphs = map(lambda p: re.sub(r"([A-Za-z]+)-\s([A-Za-z]+)", r"\1\2", p)
                                    if isinstance(p, str) else p,
                                    _paragraphs)
        # e.g. habi- tuellement, inci- dence => habituellement, incidence

        return _paragraphs


if __name__ == "__main__":
    main(get_args())
