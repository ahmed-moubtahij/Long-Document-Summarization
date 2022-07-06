from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Any, Literal, TypeAlias
from collections.abc import Iterator, Callable
import re
from itertools import dropwhile, takewhile, groupby
import docx
from simplify_docx import simplify

Text: TypeAlias = str
TextOrTable: TypeAlias = Text | list[dict[str, Any]]
Marker: TypeAlias = Literal["start_marker", "chapter_marker",
                            "header_marker", "end_marker"]

class BookLoader: # pylint: disable=too-few-public-methods

    start_marker = re.compile(r"^Introduction$")
    chapter_marker = re.compile(r"^Chapitre (\d+) \/$")
    header_marker = re.compile(r"^Chapitre \d+ \/.*")
    end_marker = re.compile(r"^Conclusion$")

    def __init__(self, data_path: Path,
                 title="Stress, santé et performance au travail",
                 markers: dict[Marker, re.Pattern[str]] | None=None):

        self.title = title
        self._paragraphs = self._parse_paragraphs(data_path)

        if markers is not None:
            self.__dict__.update(markers) # type: ignore[arg-type]

        self.chapters = self._segment_chapters()

    def _chapter_indexer(self) -> Callable[[TextOrTable], int]:

        current_chapter = 0
        def indexer(paragraph: TextOrTable) -> int:
            nonlocal current_chapter
            if isinstance(paragraph, Text):
                if found_chapter := self.chapter_marker.search(paragraph):
                    current_chapter = int(found_chapter.group(1))
            return current_chapter

        return indexer

    def _strip_headers(self, paragraphs: Iterator[TextOrTable]) -> Iterator[TextOrTable]:

        return (p for p in paragraphs
                if not isinstance(p, str)
                   or p != self.title and not self.header_marker.match(p))

    def _segment_chapters(self) -> list[list[TextOrTable]]:

        _chapters = groupby(self._paragraphs, self._chapter_indexer())

        return [list(self._strip_headers(paragraphs))
                for _, paragraphs in _chapters]

    def _parse_paragraphs(self, data_path) -> Iterator[TextOrTable]:

        assert data_path.exists()
        simple_docx = simplify(docx.Document(data_path),
                               {"include-paragraph-indent": False,
                                "include-paragraph-numbering": False})

        document: Iterator[list[dict[str, TextOrTable]]] = (
            p["VALUE"] for p in simple_docx["VALUE"][0]["VALUE"]
            if p["VALUE"] != "[w:sdt]")

        # Extract text, otherwise preserve object e.g. table
        _paragraphs: Iterator[TextOrTable] = map(
            lambda p: p[0]["VALUE"] if p[0]["TYPE"] == "text" else p, document)

        def seeking_marker(marker: re.Pattern[str]) -> Callable[[TextOrTable], bool]:
            return lambda p: not isinstance(p, str) or not marker.match(p)

        lower_bounded = dropwhile(seeking_marker(self.start_marker), _paragraphs)
        upper_bounded = takewhile(seeking_marker(self.end_marker), lower_bounded)

        # TODO: There was an "interpré- tation" in the summary output.
        # e.g. habi- tuellement => habituellement
        clean_paragraphs: Iterator[TextOrTable] = map(
            (lambda p: re.sub(r"([A-Za-z]+)-\s([A-Za-z]+)", r"\1\2", p)
             if isinstance(p, str) else p), upper_bounded)

        return clean_paragraphs


def get_args():
    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-f', "--data-fp", type=str,
                            default="data/D5627-Dolan.docx",
                            help=("Path to the docx file to summarize."))
    return arg_parser.parse_args()


def main(args) -> None:
    data_path = Path(args.data_fp).expanduser().resolve()
    book = BookLoader(data_path)
    print(book.chapters[0])


if __name__ == "__main__":
    main(get_args())
