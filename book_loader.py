from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, TypedDict
from collections.abc import Iterator, Iterable, Callable
import re
from itertools import groupby
import more_itertools as mit
import funcy as fy
import docx
from simplify_docx import simplify

T = TypeVar('T')
def unique_if(pred: Callable[[T], Any]) -> Callable[[Iterable[T]], Iterator[T]]:

    def _unique_if(iterable):
        seen = set()
        for item in iterable:
            if pred(item):
                if not item in seen:
                    seen.add(item)
                    yield item
            else:
                yield item

    return _unique_if

map_: Callable = fy.curry(map)
filter_: Callable = fy.curry(filter)
exactly_one: Callable = mit.one
values_of: Callable = partial(fy.pluck, "VALUE")

Pattern: TypeAlias = re.Pattern[str]

class Markers(TypedDict):
    start_marker: str | Pattern
    end_marker: str | Pattern
    # TODO: start_marker and end_marker should be a tuple[str | Pattern, str | Pattern] (and TypeAlias it)
    chapter_marker: str | Pattern
    header_marker: str | Pattern
    na_span_markers: tuple[str | Pattern, str | Pattern]

class BookLoader:
    start_marker: Pattern
    end_marker: Pattern
    chapter_marker: Pattern
    header_marker: Pattern
    na_span_markers: tuple[Pattern, Pattern]

    word_bisection = re.compile(r"([A-Za-z]+)-\s([A-Za-z]+)")

    def __init__(self, doc_path: Path, markers: Markers):

        assert doc_path.exists()

        _markers = fy.omit(markers, "na_span_markers")
        compiled_markers = fy.walk_values(re.compile, _markers)
        compiled_markers["na_span_markers"] = tuple(
            re.compile(m) for m in markers["na_span_markers"])
        self.__dict__.update(compiled_markers)

        self._paragraphs = self._etl_paragraphs(doc_path)
        self.chapters = self._segment_chapters()

    def _chapter_indexer(self) -> Callable[[str], int]:

        current_chapter = 0

        def indexer(paragraph):
            nonlocal current_chapter
            current_chapter += bool(self.chapter_marker.match(paragraph))
            return current_chapter

        return indexer

    def _segment_chapters(self) -> list[list[str]]:

        chapters = [list(paragraphs) for _, paragraphs in
                    groupby(self._paragraphs, self._chapter_indexer())]

        return chapters

    def _seeking_bounds(self, paragraph: str) -> bool:

        return (not self.start_marker.match(paragraph)
                and not self.end_marker.match(paragraph))

    def _is_valid_span(self) -> Callable[[str], bool]:

        valid = True

        def validator(paragraph):
            nonlocal valid
            bound_marker = self.na_span_markers[1 - valid]
            if bound_marker.match(paragraph):
                valid = not valid
            return valid

        return validator

    def _etl_paragraphs(self, doc_path: Path) -> Iterator[str]:

        process = fy.rcompose(
            self.extract_paragraphs,
            partial(mit.strip, pred=self._seeking_bounds),
            unique_if(self.header_marker.match),
            filter_(self._is_valid_span()),
            map_(partial(self.word_bisection.sub, r"\1\2")))

        return process(doc_path)


    @staticmethod
    def extract_paragraphs(doc_path: Path) -> Iterator[str]:

        _simple_docx = simplify(docx.Document(doc_path),
                               {"include-paragraph-indent": False,
                                "include-paragraph-numbering": False})
        simple_docx = exactly_one(_simple_docx["VALUE"])["VALUE"]

        extract = fy.rcompose(
            values_of,
            fy.rpartial(fy.without, "[w:sdt]"),
            map_(partial(fy.lremove, lambda u: u["TYPE"] == "CT_Empty")),
            fy.compact,
            map_(fy.iffy(pred=lambda p: p[0]["TYPE"] == "text",
                           action=lambda p: exactly_one(p)["VALUE"],
                           default=BookLoader.table_to_text)))

        return extract(simple_docx)

    @staticmethod
    def table_to_text(table: list[dict[str, Any]]) -> str:

        assert all(e["TYPE"] == "table-row" for e in table)

        return ' '.join(
            text
            for row in values_of(table)
            for cell in values_of(row)
            for content in values_of(cell)
            for text in values_of(content))


def get_args():

    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-f', "--data-fp", type=str,
                            default="data/D5627-Dolan.docx",
                            help=("Path to the docx file to summarize."))

    return arg_parser.parse_args()


def main(args) -> None:

    doc_path = Path(args.data_fp).expanduser().resolve()

    start_marker = r"^Introduction$"
    compiled_end_marker = re.compile(r"^Annexe /$")
    conclusion_marker = r"^Conclusion$"
    compiled_header_marker = re.compile(
        rf"^Chapitre \d+ /.+"
        rf"|{start_marker}"
        rf"|^Stress, santé et performance au travail$"
        rf"|{conclusion_marker}")
    chapter_marker = rf"^Chapitre \d+ /$|{conclusion_marker}"
    na_span_markers = (
            r"^exerCiCe \d\.\d /$",
            '|'.join([chapter_marker,
                      r"^Les caractéristiques personnelles\.",
                      r"/\tLocus de contrôle$",
                      r"^L'observation de sujets a amené Rotter",
                      r"^Lorsqu'une personne souffre de stress"]))

    book = BookLoader(doc_path,
                      {"start_marker": start_marker,
                       "end_marker": compiled_end_marker,
                       "chapter_marker": chapter_marker,
                       "header_marker": compiled_header_marker,
                       "na_span_markers": na_span_markers})

    expected_lengths = [44, 136, 194, 178, 345, 348, 29]
    assert [len(c) for c in book.chapters] == expected_lengths

if __name__ == "__main__":
    main(get_args())
