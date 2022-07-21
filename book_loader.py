from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Literal, Tuple, TypeAlias
from collections.abc import Iterator, Callable
import re
from itertools import groupby
from more_itertools import one, split_at, strip
from funcy import rcompose, iffy
import docx
from simplify_docx import simplify

class BookLoader:

    Table: TypeAlias = list[dict[str, Any]]
    TextOrTable: TypeAlias = str | Table
    MarkerKey: TypeAlias = Literal["start_marker", "chapter_marker",
                                   "header_marker", "end_marker",
                                   "ps_marker", "na_span_markers"]
    Marker: TypeAlias = re.Pattern[str] | Tuple[list[re.Pattern[str]], list[re.Pattern[str]]]

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

    def _chapter_indexer(self) -> Callable[[str], int]:

        current_chapter = 0

        def indexer(paragraph):
            nonlocal current_chapter
            if found_chapter := self.chapter_marker.search(paragraph):
                current_chapter = int(found_chapter.group(1)) # type: ignore[union-attr]

            return current_chapter

        return indexer

    def _is_not_header(self) -> Callable[[str], bool]:

        is_start = True

        def not_header(paragraph):
            nonlocal is_start
            header_match = self.header_marker.match(paragraph)
            if is_start and header_match:
                is_start = False
                return True

            return not header_match

        return not_header

    def _segment_chapters(self) -> list[list[str]]:

        chapters = [list(paragraphs) for _, paragraphs in
                    groupby(self._paragraphs, self._chapter_indexer())]

        last_chapter, separator, post_scriptum = split_at(
            chapters[-1], self.ps_marker.match, maxsplit=1, keep_separator=True)
        chapters[-1] = last_chapter
        chapters.extend([separator + post_scriptum])

        return chapters

    def _seeking_bounds(self, paragraph: str) -> bool:

        return (not self.start_marker.match(paragraph)
                and not self.end_marker.match(paragraph))

    def _spans_validator(self) -> Callable[[str], bool]:

        valid = True

        def validator(paragraph):
            nonlocal valid
            bound_markers = self.na_span_markers[1 - valid]
            if any(map(lambda pat: pat.match(paragraph), bound_markers)):
                valid = not valid

            return valid

        return validator

    def _etl_paragraphs(self, data_path: Path) -> Iterator[str]:

        transform = rcompose(
            partial(strip, pred=self._seeking_bounds),
            partial(filter, self._is_not_header()),
            partial(filter, self._spans_validator()),
            partial(map, self.join_bisected_words))

        return transform(self.extract_paragraphs(data_path))

    @staticmethod
    def table_to_text(paragraph: Table) -> str:

        assert all(e["TYPE"] == "table-row" for e in paragraph)
        values_of: Callable = partial(map, itemgetter("VALUE"))

        return ' '.join(
            text
            for row in values_of(paragraph)
            for cell in values_of(row)
            for content in values_of(cell)
            for text in values_of(content))

    @staticmethod
    def join_bisected_words(paragraph: str) -> str:

        return re.sub(r"([A-Za-z]+)-\s([A-Za-z]+)", r"\1\2", paragraph)

    I: TypeAlias = Iterator[list[dict[str, TextOrTable]]]
    @staticmethod
    def remove_empty_content(doc: I) -> I:

        doc = filter(lambda p: p != "[w:sdt]", doc)
        doc = map(lambda p: [u for u in p if u["TYPE"] != "CT_Empty"], doc)
        doc = filter(bool, doc)

        return doc

    @staticmethod
    def extract_paragraphs(data_path: Path) -> Iterator[TextOrTable]:

        assert data_path.exists()
        simple_docx = simplify(docx.Document(data_path),
                               {"include-paragraph-indent": False,
                                "include-paragraph-numbering": False})
        doc_values = map(itemgetter("VALUE"), one(simple_docx["VALUE"])["VALUE"])
        doc = BookLoader.remove_empty_content(doc_values)
        paragraphs = map(iffy(pred=lambda p: p[0]["TYPE"] == "text",
                              action=lambda p: one(p)["VALUE"],
                              default=BookLoader.table_to_text),
                         doc)

        return paragraphs


def get_args():

    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-f', "--data-fp", type=str,
                            default="data/D5627-Dolan.docx",
                            help=("Path to the docx file to summarize."))

    return arg_parser.parse_args()


def main(args) -> None:

    data_path = Path(args.data_fp).expanduser().resolve()
    book = BookLoader(data_path)
    assert [len(c) for c in book.chapters] == [43, 135, 193, 177, 344, 347, 31]


if __name__ == "__main__":
    main(get_args())
