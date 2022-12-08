from collections.abc import Callable, Iterator, Iterable
from typing import ClassVar, Literal
from itertools import islice, chain, zip_longest, pairwise
from functools import partial

import deal
import numpy as np
import scipy.spatial.distance as spd
from sentence_transformers import SentenceTransformer
import networkx as nx

from LDS.ext_summarizers import ExtractiveSummarizer
from LDS.gen_utils import UnaryPred, stable_unique_list, identity_f
from LDS.nlp_utils import french_sentencizer, SentenceEncoderProto

class TextRank(ExtractiveSummarizer):

    SentenceEncoder = Literal[
        "camembert",
        "french_semantic",
        "hugorosen_flaubert",
        "inokufu_flaubert",
        "flaubert_education"
    ] | SentenceEncoderProto

    accept_all: ClassVar[UnaryPred[str]] = lambda _: True

    @deal.pre(lambda _: 0 <= _.damping <= 1)
    @deal.pre(lambda _: 0 <= _.sim_threshold <= 1)
    @deal.pre(lambda _: 0 <= _.convergence_threshold <= 1)
    @deal.pre(lambda _: _.max_iterations > 0)
    def __init__(self,
        sentence_encoder:          SentenceEncoder,
        use_eigen_solver_pagerank: bool                 = True,
        damping:                   float                = 0.85, # (Mihalcea & Tarau, 2004)
        convergence_threshold:     float                = 1e-4, # (Mihalcea & Tarau, 2004)
        sim_threshold:             float                = 0.65, # (Kazemi et al. 2020)
        max_iterations:            int                  = 100,  # (Florescu et al. 2017)
        sentence_pred:             UnaryPred[str]       = accept_all,
        paragraph_gap:             int                  = 4,
        post_process:              Callable[[str], str] = identity_f,
    ):
        print("Using sentence encoder:", sentence_encoder)
        _sentence_encoder = SentenceTransformer(
            {  # Convenient mappings with fallback to the passed encoder
               "camembert":
                    "dangvantuan/sentence-camembert-large",
               "french_semantic":
                    "Sahajtomar/french_semantic",
               "hugorosen_flaubert":
                    "hugorosen/flaubert_base_uncased-xnli-sts",
               "inokufu_flaubert":
                    "inokufu/flaubert-base-uncased-xnli-sts",
               "flaubert_education":
                   "inokufu/flaubert-base-uncased-xnli-sts-finetuned-education",
            }.get(sentence_encoder, sentence_encoder) # type: ignore[arg-type]
        )
        self.encode        = _sentence_encoder.encode # normalize_embeddings + dotp = worse scores
        self.sim_threshold = sim_threshold
        self.sentence_pred = sentence_pred
        self.paragraph_gap = paragraph_gap
        self.pagerank      = (partial(nx.pagerank_numpy, alpha=damping)
                              if use_eigen_solver_pagerank else
                              partial(nx.pagerank,
                                      alpha=damping,
                                      tol=convergence_threshold,
                                      max_iter=max_iterations))
        self.post_process  = post_process

    @deal.ensure(lambda _: len(_.result) <= len(_.text))
    def __call__(self, text: str, n_sentences: int) -> str:

        super().__call__(text, n_sentences)
        # Duplicate sentences are unlikely, but such an event spuriously boosts centrality.
        sentences = stable_unique_list(s.strip() for s in french_sentencizer(text)
                                       if self.sentence_pred(s))
        ranked_sentences = self.textrank(sentences)
        top_sentences = list(self.__class__.top_n_in_order(ranked_sentences, n_sentences))
        joined_summary = self.pos_based_join(
            zip(top_sentences, map(sentences.index, top_sentences))
        )
        summary = self.post_process(joined_summary)

        return summary

    def textrank(self, sentences: list[str]) -> dict[str, float]:

        adj_mat = spd.squareform(
            1 - spd.pdist(self.encode(sentences), metric="cosine")
        ).astype(np.float16)
        adj_mat[adj_mat < self.sim_threshold] = 0

        textrank_graph = nx.relabel_nodes(
            nx.from_numpy_array(adj_mat),
            dict(enumerate(sentences)), # index-nodes to sentence-nodes
            copy=False
        )
        return self.pagerank(textrank_graph)

    def pos_based_join(self, text_segments_with_pos: Iterable[tuple[str, int]]) -> str:

        text_segs, positions = zip(*text_segments_with_pos)
        separators = [' ' if abs(p2 - p1) < self.paragraph_gap # pyright: ignore
                      else '\n'
                      for p1, p2 in pairwise(positions)]

        return ''.join(chain.from_iterable( # pyright: ignore
            zip_longest(text_segs, separators, fillvalue='')
        ))

    @staticmethod
    @deal.pre(lambda _: _.n > 0)
    def top_n_in_order(ranked_texts: dict[str, float], n: int) -> Iterator[str]:

        nth_best_rank = np.partition([*ranked_texts.values()], -n)[-n]

        return islice((text for text, rank in ranked_texts.items()
                       if rank >= nth_best_rank), n)
