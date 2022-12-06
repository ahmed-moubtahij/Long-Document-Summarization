import warnings
from pathlib import Path
import re
import json
from collections.abc import Iterator, Callable
from typing import ClassVar, TypeAlias, TypedDict
from functools import partial

from typing_extensions import Required
import funcy as fy
from funcy_chain import IterChain, Chain
import docx
from simplify_docx import simplify
import deal

import LDS.gen_utils as ut
from LDS.nlp_utils import match_anything, match_nothing
from LDS.nlp_utils import RE_ALPHA

warnings.filterwarnings("ignore", message="Skipping unexpected tag")

RgxPattern:     TypeAlias = re.Pattern[str]
Marker:         TypeAlias = RgxPattern | str
MarkersPair:    TypeAlias = list[str | RgxPattern]
PatternsPair:   TypeAlias = list[RgxPattern]

class Markers(TypedDict, total=False):
    chapter:        Required[Marker]
    slice:          MarkersPair
    headers:        Marker
    footnotes:      Marker
    undesirables:   Marker
    citing:         Marker
    na_span:        MarkersPair

class BookLoader:
    chapter:        RgxPattern
    """Matched for grouping paragraphs by their respective chapters, and for joining newlines."""
    slice:          PatternsPair = [match_anything, match_anything]
    """Pair to consecutively match for slicing the document."""
    headers:        RgxPattern = match_nothing
    """Matched for removing duplicate headers, and for joining newlines."""
    footnotes:      RgxPattern = match_nothing
    """Matched for removing footnotes."""
    undesirables:   RgxPattern = match_nothing
    """Matched for removing miscellaneous undesirables."""
    citing:         RgxPattern = match_nothing
    """Substituted with its number-stripped self.
       e.g. "aa cited1 bb cited2 cc" -> "aa cited bb cited cc"
    """
    na_span:        PatternsPair = [match_nothing, match_nothing]
    """Matched from start to end for removing non-applicable spans.
       e.g. removing recurring practice sections from a pedagogical book.
    """
    chapters:       list[str]
    """List of cleaned and grouped chapters."""

    bisection:      ClassVar = re.compile(
        rf"""({RE_ALPHA}+)  # Captured 1st part of the bisected word
             -\s            # bisection
             ({RE_ALPHA}+)  # Captured 2nd part of the bisected word.
        """, re.VERBOSE)
    """ Substituted with its bisected parts. Such bisections oc-
        cur at the end of lines. e.g. 'cri- tique' -> 'critique'.
    """
    within_parens:  ClassVar = re.compile(
        r"""\(.*?           # Opening paren followed by 0 or more characters
            \..*?           # A dot followed by 0 or more characters
            \)              # Closing paren.
        """, re.VERBOSE)
    """ Matched to remove dots from within parens to avoid corrupting sentencization.
        e.g. 'aa (p. ex. bb) cc.' -> 'aa (p ex bb) cc.'
    """
    params_json_contract: ClassVar = deal.chain(
        deal.pre(lambda _: _.params_json.exists()),
        deal.pre(lambda _: _.params_json.is_file()),
        deal.pre(lambda _: str(_.params_json).endswith('.json')),
        deal.has('read')
    )
    get_chapters_contract = deal.chain(
        deal.pre(lambda _: _.first >= 0),
        deal.pre(lambda _: _.last >= _.first),
        deal.pre(lambda _: len(_.self.chapters) >= _.last - _.first + 2),
        deal.pure,
        deal.ensure(lambda _: len(_.result) == _.last - _.first + 1)
    )
    def __init__(self, doc_path: str, markers: Markers) -> None:

        self.doc_path = Path(doc_path).resolve()

        _compiled_markers = fy.walk_values(
            fy.iffy(pred=fy.is_list,
                    action=ut.lmap_(re.compile),
                    default=re.compile),
            markers
        )
        self.__dict__.update(_compiled_markers)

        _paragraphs = self._etl_paragraphs()
        self.chapters = (Chain(_paragraphs)
                            .group_by(self._chapter_indexer())
                            .values()
                            .map(ut.reduce_(self._with_newline_or_space))
                        ).value

    @classmethod
    @params_json_contract
    def from_params_json(cls, params_json=Path("parameters.json")):

        return cls(**json.loads(
            params_json.read_text(encoding="utf-8")
        ))

    @deal.ensure(lambda _: len(_.result) ==
                           len(_.l_paragraph) + 1 + len(_.r_paragraph))
    def _with_newline_or_space(
        self, l_paragraph: str, r_paragraph: str
    ) -> str:
        joiner = ('\n' if l_paragraph.endswith('.')
                          or self.headers.match(l_paragraph)
                          or self.chapter.match(l_paragraph)
                  else ' ')
        return l_paragraph + joiner + r_paragraph

    @get_chapters_contract
    def get_chapters(
        self, first=0, last: int | None=None, skip_headers=True
    ) -> list[str]:
        """ Reads chapters `first` to `last` inclusively with optionally skipped headers """
        chapters = self.chapters[first: None if last is None
                                             else last + 1]
        if skip_headers:
            return [ # Assumes '\n' at the end of a chapter's header
                chapter[chapter.index('\n') + 1: ]
                for chapter in chapters
            ]

        return chapters

    def _chapter_indexer(self) -> Callable[[str], int]:

        current_chapter = 0

        @deal.has('global')
        def indexer(paragraph):
            nonlocal current_chapter
            current_chapter += bool(self.chapter.match(paragraph))
            return current_chapter

        return indexer

    def _is_valid_span(self) -> Callable[[str], bool]:

        valid = True

        @deal.has('global')
        def validator(paragraph):
            nonlocal valid
            bound_marker = self.na_span[1 - valid]
            if bound_marker.match(paragraph):
                valid = not valid
            return valid

        return validator

    def _etl_paragraphs(self) -> Iterator[str]:

        paragraphs = read_paragraphs(self.doc_path)

        return (IterChain(paragraphs)
                    .thru(ut.strip_(fy.none_fn(*self.slice)))
                    .thru(ut.unique_if_(self.headers.match))
                    .filter(self._is_valid_span())
                    .remove(self.footnotes)
                    .remove(self.undesirables)
                    .map(partial(self.bisection.sub, r'\1\2'))
                    .map(partial(self.citing.sub, r'\1'))
                    .map(partial(self.within_parens.sub, rm_match_dots))
               ).value

@deal.has()
def rm_match_dots(match: re.Match) -> str:
    return match.group(0).replace('.', '')

@deal.has()
@deal.pre(lambda table: all(e["TYPE"] == "table-row" for e in table))
def table_to_text(table: list[dict[str, list[dict]]]) -> str:

    values_of = partial(fy.pluck, "VALUE")

    return ' '.join(
        text
        for row in values_of(table)
        for cell in values_of(row)
        for content in values_of(cell)
        for text in values_of(content)
    )

read_paragraphs_contract = deal.chain(
    deal.has('read'),
    deal.pre(lambda doc_path: doc_path.exists()),
    deal.pre(lambda doc_path: doc_path.is_file()),
    deal.pre(lambda doc_path: str(doc_path).endswith('.docx'))
)
@read_paragraphs_contract
def read_paragraphs(doc_path: Path) -> Iterator[str]:

    _simple_docx = simplify(
        docx.Document(doc_path),
        {"include-paragraph-indent": False,
         "include-paragraph-numbering": False}
    )
    simple_docx = ut.one_expected(_simple_docx["VALUE"])["VALUE"]

    return (IterChain(simple_docx)
                .pluck("VALUE")
                .without("[w:sdt]")
                .map(ut.lwhere_not_(TYPE="CT_Empty"))
                .compact()
                .map(fy.iffy(pred=lambda p: p[0]["TYPE"] == "text",
                             action=lambda p: ut.one_expected(p)["VALUE"],
                             default=table_to_text))
            ).value
