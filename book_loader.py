from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from operator import itemgetter
from pathlib import Path
from typing import Any, Literal, Tuple, TypeAlias
from collections.abc import Iterator, Callable
import re
from itertools import groupby
from more_itertools import strip, split_at, one
import docx
from simplify_docx import simplify

Text: TypeAlias = str
Table: TypeAlias = list[dict[str, Any]]
TextOrTable: TypeAlias = Text | Table
MarkerKey: TypeAlias = Literal["start_marker", "chapter_marker",
                               "header_marker", "end_marker",
                               "ps_marker", "na_span_markers"]
Marker: TypeAlias = re.Pattern[str] | Tuple[list[re.Pattern[str]], list[re.Pattern[str]]]

class BookLoader:

    start_marker = re.compile(r"^Introduction$")
    chapter_marker = re.compile(r"^Chapitre (\d+) /$")
    header_marker = re.compile(rf"(?:^Chapitre \d+ /.+"
                               rf"|{start_marker.pattern}"
                               rf"|^Stress, santé et performance au travail$)")
    end_marker = re.compile(r"^Annexe /$")
    ps_marker = re.compile(r"^Conclusion$")
    na_span_markers = (
        [re.compile(r"^exerCiCe \d\.\d /$")],
        [chapter_marker,
         re.compile(r"^Les caractéristiques personnelles\."),
         re.compile(r"/\tLocus de contrôle$"),
         re.compile(r"^L'observation de sujets a amené Rotter"),
         re.compile(r"^Lorsqu'une personne souffre de stress")])

    def __init__(self, data_path: Path,
                 markers: dict[MarkerKey, Marker] | None=None):

        self._paragraphs = self._etl_paragraphs(data_path)

        if markers is not None:
            self.__dict__.update(markers) # type: ignore[arg-type]

        self.chapters = self._segment_chapters()

    def _chapter_indexer(self) -> Callable[[TextOrTable], int]:

        current_chapter = 0

        def indexer(paragraph: TextOrTable) -> int:
            nonlocal current_chapter
            if isinstance(paragraph, Text):
                if found_chapter := self.chapter_marker.search(paragraph):
                    current_chapter = int(found_chapter.group(1)) # type: ignore[union-attr]
            return current_chapter

        return indexer

    def _is_not_header(self) -> Callable[[TextOrTable], bool]:

        is_start = True

        def not_header(paragraph):
            nonlocal is_start
            header_match = self.__class__.match(self.header_marker, paragraph)
            if is_start and header_match:
                is_start = False
                return True

            return not header_match

        return not_header

    def _segment_chapters(self) -> list[list[TextOrTable]]:

        chapters = [list(paragraphs) for _, paragraphs in
                    groupby(self._paragraphs, self._chapter_indexer())]

        last_chapter, separator, post_scriptum = split_at(
            chapters[-1], lambda _p: self.__class__.match(self.ps_marker, _p),
            maxsplit=1, keep_separator=True)
        chapters[-1] = last_chapter
        chapters.extend([separator + post_scriptum])

        return chapters

    def _seeking_bounds(self, paragraph: TextOrTable) -> bool:

        return (not self.__class__.match(self.start_marker, paragraph)
                and not self.__class__.match(self.end_marker, paragraph))

    def _spans_validator(self) -> Callable[[TextOrTable], bool]:

        valid = True

        def validator(paragraph):
            nonlocal valid
            if isinstance(paragraph, Text):
                bound_markers = self.na_span_markers[1 - valid]
                if any(map(lambda pat: pat.match(paragraph), bound_markers)):
                    valid = not valid

            return valid

        return validator

    def _etl_paragraphs(self, data_path: Path) -> Iterator[TextOrTable]:

        paragraphs = self.__class__.extract_paragraphs(data_path)
        bounded_doc = strip(paragraphs, self._seeking_bounds)
        headless_paragraphs = filter(self._is_not_header(), bounded_doc)
        valid_paragraphs = filter(self._spans_validator(), headless_paragraphs)
        clean_paragraphs = map(self.__class__.join_bisected_words, valid_paragraphs)

        # TODO: Extract tables contents
        # tables = filter(lambda p: not isinstance(p, str), clean_paragraphs)
        # tables = list(tables)

        return clean_paragraphs

    @staticmethod
    def match(marker: re.Pattern[str], paragraph: TextOrTable) -> bool:

        return isinstance(paragraph, str) and bool(marker.match(paragraph))

    @staticmethod
    def join_bisected_words(paragraph: TextOrTable) -> TextOrTable:

        if not isinstance(paragraph, str):
            return paragraph

        return re.sub(r"([A-Za-z]+)-\s([A-Za-z]+)", r"\1\2", paragraph)

    I: TypeAlias = Iterator[list[dict[str, TextOrTable]]]
    @staticmethod
    def remove_ct_empty(doc: I) -> I:

        doc = map(lambda p: [u for u in p if u["TYPE"] != "CT_Empty"], doc)

        return filter(bool, doc)

    @staticmethod
    def extract_paragraphs(data_path: Path) -> Iterator[TextOrTable]:

        assert data_path.exists()
        simple_docx = simplify(docx.Document(data_path),
                               {"include-paragraph-indent": False,
                                "include-paragraph-numbering": False})
        _doc = map(itemgetter("VALUE"), simple_docx["VALUE"][0]["VALUE"])
        doc = BookLoader.remove_ct_empty(filter(lambda p: p != "[w:sdt]", _doc))

        return map(lambda p: one(p)["VALUE"] if p[0]["TYPE"] == "text" else p, doc)


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
