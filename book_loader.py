from functools import partial
from pathlib import Path
from typing import TypeAlias, TypedDict
from collections.abc import Iterator, Callable
import re
import json
import warnings
from more_itertools import strip
import funcy as fy
from funcy_chain import IterChain, Chain
import docx
from simplify_docx import simplify
import my_utils as ut
warnings.filterwarnings("ignore", message="Skipping unexpected tag")

values_of: Callable = partial(fy.pluck, "VALUE")

Pattern: TypeAlias = re.Pattern[str]
Marker: TypeAlias = Pattern | str
MarkersPair: TypeAlias = list[str | Pattern]
PatternsPair: TypeAlias = list[Pattern]

class Markers(TypedDict):
    slice: MarkersPair
    chapter: Marker
    header: Marker
    references: Marker
    undesirables: Marker
    na_span: MarkersPair

class BookLoader:
    doc_path: Path
    slice: PatternsPair
    chapter: Pattern
    header: Pattern
    references: Pattern
    undesirables: Pattern
    na_span: PatternsPair
    chapters: list[list[str]]

    re_compile = partial(re.compile, flags=re.UNICODE)

    def __init__(self, doc_path: str, markers: Markers):

        self.doc_path = Path(doc_path).expanduser().resolve()
        assert self.doc_path.exists()

        _compiled_markers = fy.walk_values(
            fy.iffy(pred=fy.is_list,
                    action=ut.lmap_(self.re_compile),
                    default=self.re_compile),
            markers)
        self.__dict__.update(_compiled_markers)

        _paragraphs = self._etl_paragraphs()
        self.chapters = (Chain(_paragraphs)
                            .group_by(self._chapter_indexer())
                            .values()
                            .map(list)
                        ).value

    def _chapter_indexer(self) -> Callable[[str], int]:

        current_chapter = 0

        def indexer(paragraph):
            nonlocal current_chapter
            current_chapter += bool(self.chapter.match(paragraph))
            return current_chapter

        return indexer

    def _is_valid_span(self) -> Callable[[str], bool]:

        valid = True

        def validator(paragraph):
            nonlocal valid
            bound_marker = self.na_span[1 - valid]
            if bound_marker.match(paragraph):
                valid = not valid
            return valid

        return validator

    def _etl_paragraphs(self) -> Iterator[str]:

        paragraphs = self.extract_paragraphs(self.doc_path)

        return (IterChain(paragraphs)
                    .thru(partial(strip, pred=fy.none_fn(*self.slice)))
                    .thru(ut.unique_if_(self.header.match))
                    .filter(self._is_valid_span())
                    .remove(self.references)
                    .remove(self.undesirables)
                    .thru(self.join_bisections())
               ).value

    @staticmethod
    def join_bisections() -> Callable[[str], Iterator[str]]:

        bisection = re.compile(r"(\w+)-\s(\w+)", re.UNICODE)

        return lambda paragraph: map(
            partial(bisection.sub, r"\1\2"), paragraph)

    @staticmethod
    def extract_paragraphs(doc_path: Path) -> Iterator[str]:

        _simple_docx = simplify(docx.Document(doc_path),
                               {"include-paragraph-indent": False,
                                "include-paragraph-numbering": False})
        simple_docx = ut.exactly_one(_simple_docx["VALUE"])["VALUE"]

        return (IterChain(simple_docx)
                    .thru(values_of)
                    .without("[w:sdt]")
                    .map(ut.lwhere_not_(TYPE="CT_Empty"))
                    .compact()
                    .map(fy.iffy(pred=lambda p: p[0]["TYPE"] == "text",
                                 action=lambda p: ut.exactly_one(p)["VALUE"],
                                 default=BookLoader.table_to_text))
               ).value

    @staticmethod
    def table_to_text(table: list[dict[str, list[dict]]]) -> str:

        assert all(e["TYPE"] == "table-row" for e in table)

        return ' '.join(
            text
            for row in values_of(table)
            for cell in values_of(row)
            for content in values_of(cell)
            for text in values_of(content))


def main() -> None:

    with open('parameters.json', 'r', encoding="utf-8") as json_file:
        params = json.load(json_file)

    book = BookLoader(**params)

    observed_lengths = [len(c) for c in book.chapters]
    expected_lengths = [43, 101, 152, 136, 271, 307, 23]

    assert observed_lengths == expected_lengths

if __name__ == "__main__":
    main()
