from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from pathlib import Path
from typing import Any, Iterator, List, Union, Dict, Callable, Literal, TypeAlias
import re
from itertools import chain, dropwhile, groupby
import docx
from simplify_docx import simplify


def main(args) -> None:
    data_path = make_data_path(args.data_dir, args.data_fn)
    book = BookLoader(data_path)
    print(book.chapters[0])


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


# pylint: disable=no-member # `setattr` dynamic members are opaque to type checkers
class BookLoader:
# TODO: --too-few-public-methods. As a data loader, Should this be a dataclass? Look into their usecases

    Table: TypeAlias = List[dict]
    TextOrTable: TypeAlias = Union[str, Table]
    Marker: TypeAlias = Literal["intro_marker", "chapter_marker", "header_marker"]

    default_markers: Dict[Marker, re.Pattern[str]] = {
        "intro_marker": re.compile(r"^Introduction$"),
        "chapter_marker": re.compile(r"^Chapitre (\d+) \/$"),
        "header_marker": re.compile(r"^Chapitre \d+ \/.*")
    }

    def __init__(self, data_path: Path, title="",
                 markers: Dict[Marker, re.Pattern[str]] | None=None):

        assert data_path.exists()
        self.docx: dict = simplify(docx.Document(data_path))
        self.paragraphs = self._init_paragraphs()
        self.title = title if title else next(self.paragraphs)

        if markers is None:
            markers = {}
        for marker, default_pattern in BookLoader.default_markers.items():
            setattr(self, marker, markers.get(marker, default_pattern))

        self.chapters = self._init_chapters()
        # TODO: Have a property getter for self.chapters.
        # This could help with pylint's `too-few-public-methods`

    def _chapter_indexer(self) -> Callable[[TextOrTable], int]:

        current_chapter = 0
        def indexer(paragraph: Union[str, List[dict]]) -> int:
            nonlocal current_chapter
            if isinstance(paragraph, str):
                if found_chapter := self.chapter_marker.search(paragraph):
                    current_chapter = int(found_chapter.group(1))

            return current_chapter

        return indexer

    def _strip_headers(self, paragraphs: Iterator[TextOrTable]) -> Iterator[TextOrTable]:

        return (p for p in paragraphs
                if not isinstance(p, str) # not str => Table => not a header
                or (p != self.title
                    and not self.intro_marker.match(p)
                    and not self.header_marker.match(p)))

    def _init_chapters(self) -> Dict[int, List[TextOrTable]]:

        paragraphs_from_intro = dropwhile(
            lambda p: not isinstance(p, str) or not self.intro_marker.match(p),
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
        _paragraphs = map(lambda p: re.sub(r"([A-Za-z]+)-\s([A-Za-z]+)", r"\1\2", p)\
                                    if isinstance(p, str) else p,\
                                    _paragraphs) # e.g. habi- tuellement => habituellement


        return _paragraphs


if __name__ == "__main__":

    main(get_args())
