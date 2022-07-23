from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias, TypedDict
from collections.abc import Iterator, Callable
import re
from itertools import groupby
import more_itertools as mit
import funcy as fy
import docx
from simplify_docx import simplify

values_of: Callable = partial(fy.pluck, "VALUE")
exactly_one: Callable = mit.one

Pattern: TypeAlias = re.Pattern[str]
PatternsPair: TypeAlias = tuple[list[Pattern], list[Pattern]]
StrsPair: TypeAlias = tuple[list[str], list[str]]

class Markers(TypedDict):
    start_marker: str | Pattern
    end_marker: str | Pattern
    chapter_marker: str | Pattern
    header_marker: str | Pattern
    ps_marker: str | Pattern
    na_span_markers: PatternsPair | StrsPair

class BookLoader:
    start_marker: Pattern
    end_marker: Pattern
    chapter_marker: Pattern
    header_marker: Pattern
    ps_marker: Pattern
    na_span_markers: PatternsPair

    word_bisection = re.compile(r"([A-Za-z]+)-\s([A-Za-z]+)")

    def __init__(self, data_path: Path, markers: Markers):

        assert data_path.exists()

        _markers = fy.omit(markers, "na_span_markers")
        compiled_markers = fy.walk_values(re.compile, _markers)
        compiled_markers["na_span_markers"] = tuple(
            [re.compile(m) for m in span] for span in markers["na_span_markers"])
        self.__dict__.update(compiled_markers)

        self._paragraphs = self._etl_paragraphs(data_path)
        self.chapters = self._segment_chapters()

    def _chapter_indexer(self) -> Callable[[str], int]:

        current_chapter = 0

        def indexer(paragraph):
            nonlocal current_chapter
            if found_chapter := self.chapter_marker.search(paragraph):
                current_chapter = int(found_chapter.group(1))

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

        last_chapter, separator, post_scriptum = mit.split_at(
            chapters[-1], pred=self.ps_marker.match,
            maxsplit=1, keep_separator=True)
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

        paragraphs = self.extract_paragraphs(data_path)

        transform = fy.rcompose(
            partial(mit.strip, pred=self._seeking_bounds),
            partial(filter, self._is_not_header()),
            partial(filter, self._spans_validator()),
            partial(map, partial(self.word_bisection.sub, r"\1\2")))

        return transform(paragraphs)

    @staticmethod
    def extract_paragraphs(data_path: Path) -> Iterator[str]:

        _simple_docx = simplify(docx.Document(data_path),
                               {"include-paragraph-indent": False,
                                "include-paragraph-numbering": False})
        simple_docx = exactly_one(_simple_docx["VALUE"])["VALUE"]

        extract = fy.rcompose(
            values_of,
            fy.rpartial(fy.without, "[w:sdt]"),
            partial(map, partial(fy.lremove, lambda u: u["TYPE"] == "CT_Empty")),
            fy.compact,
            partial(map, fy.iffy(pred=lambda p: p[0]["TYPE"] == "text",
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

    data_path = Path(args.data_fp).expanduser().resolve()

    start_marker = r"^Introduction$"
    compiled_header_marker = re.compile(rf"(?:^Chapitre \d+ /.+"
                               rf"|{start_marker}"
                               rf"|^Stress, santé et performance au travail$)")
    chapter_marker = r"^Chapitre (\d+) /$"
    na_span_markers = (
            [r"^exerCiCe \d\.\d /$"],
            [chapter_marker,
             r"^Les caractéristiques personnelles\.",
             r"/\tLocus de contrôle$",
             r"^L'observation de sujets a amené Rotter",
             r"^Lorsqu'une personne souffre de stress"])

    book = BookLoader(data_path,
                      {"start_marker": start_marker,
                       "end_marker": re.compile(r"^Annexe /$"),
                       "chapter_marker": chapter_marker,
                       "header_marker": compiled_header_marker,
                       "ps_marker": re.compile(r"^Conclusion$"),
                       "na_span_markers": na_span_markers})

    expected_lengths = [43, 135, 193, 177, 344, 347, 31]
    assert [len(c) for c in book.chapters] == expected_lengths

if __name__ == "__main__":
    main(get_args())
