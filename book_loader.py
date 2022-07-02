from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from pathlib import Path
# TODO: https://peps.python.org/pep-0585/
from typing import Any, Iterator, Callable, Literal, TypeAlias
import re
from itertools import chain, dropwhile, groupby
import docx
from simplify_docx import simplify


class BookLoader: # pylint: disable=too-few-public-methods
    TextOrTable: TypeAlias = str | list[dict]
    Marker: TypeAlias = Literal["intro_marker", "chapter_marker", "header_marker"]

    intro_marker = re.compile(r"^Introduction$")
    chapter_marker = re.compile(r"^Chapitre (\d+) \/$")
    header_marker = re.compile(r"^Chapitre \d+ \/.*")

    def __init__(self, data_path: Path, title="",
                 markers: dict[Marker, re.Pattern[str]] | None=None):

        assert data_path.exists()
        self._docx = simplify(docx.Document(data_path))
        self._paragraphs = self._init_paragraphs()
        self.title = title if title else next(self._paragraphs)

        if markers is not None:
            self.__dict__.update(markers) # type: ignore[arg-type]

        self.chapters = self._init_chapters()

    def _chapter_indexer(self) -> Callable[[TextOrTable], int]:

        current_chapter = 0
        def indexer(paragraph: str | list[dict]) -> int:
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

    def _init_chapters(self) -> dict[int, list[TextOrTable]]:

        paragraphs_from_intro = dropwhile(
            lambda p: not isinstance(p, str) or not self.intro_marker.match(p),
            self._paragraphs)

        _chapters = groupby(paragraphs_from_intro, self._chapter_indexer())
        chapters = {idx: list(self._strip_headers(paragraphs))
                    for idx, paragraphs in _chapters}
        return chapters

    def _init_paragraphs(self) -> Iterator[TextOrTable]:

        _docx: list[dict] = self._docx["VALUE"][0]["VALUE"]

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


def get_args() -> argparse.Namespace:
    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-d', '--data-dir', type=str,
                            default="data/",
                            help="Path to directory containing the input data file")
    arg_parser.add_argument('-f', "--data-fn", type=str,
                            default="D5627-Dolan.docx",
                            help=("File name with the text to summarize"))
    return arg_parser.parse_args()


def main(args) -> None:
    data_path = Path(args.data_dir + args.data_fn).expanduser().resolve()
    book = BookLoader(data_path)
    print(book.chapters[0])

if __name__ == "__main__":

    main(get_args())
