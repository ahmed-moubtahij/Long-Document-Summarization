from collections.abc import Iterator, Callable
from typing import ClassVar, Literal, TypeAlias
import heapq
from pathlib import Path
from operator import itemgetter

import numpy as np
import numpy.typing as npt
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from spacy.lang.fr import French
import deal

from book_loader import BookLoader
from summarizer_ios import read_references
from summarizer_ios import output_summaries
from summarizer_ios import print_sample
from nlp_utils import french_sentencizer

# TODO: Typehint & contract this

def main():

    book = BookLoader.from_params_json()

    chapters_to_summarize = book.get_chapters(1, 3)
    references = read_references(Path("data/references/").resolve())

    print("GENERATING SUMMARIES PER CHAPTER...")
    summarizer = FrenchTextRank()
    summary_units = [
        {
            "CHAPTER": idx + 1,
            "SUMMARY": summarizer(chapter,
                                  len(french_sentencizer(ref)),
                                  lambda s: len(s.split()) > 4),
            "REFERENCE": ref
        }
        for idx, (chapter, ref) in enumerate(zip(chapters_to_summarize, references))
    ]

    out_path = output_summaries(summary_units,
                                out_path=Path("data/output_summaries/").resolve(),
                                model_name="textrank")

    print_sample(out_path, just_one=False)

@deal.inv(lambda french_TR: 0.0 <= french_TR.damping_factor <= 1.0)
class FrenchTextRank():

    IndexedText:        TypeAlias = tuple[int, str]
    IndexedTexts:       TypeAlias = Iterator[IndexedText]
    Ranking:            TypeAlias = npt.NDArray[np.float64]
    RankedText:         TypeAlias = tuple[IndexedText, Ranking]
    SentenceEncoder =   Literal["camembert",
                                "french_semantic",
                                "hugorosen_flaubert",
                                "inokufu_flaubert",
                                "flaubert_education"]
    nlp: ClassVar = French()
    nlp.add_pipe("sentencizer")

    @deal.has('io')
    @deal.raises(NotImplementedError)
    def __init__(self,
                 damping_factor=0.8, # TODO: Is the damping factor correctly used?
                 sentence_encoder: SentenceEncoder = "camembert"):

        self.damping_factor = damping_factor

        match sentence_encoder:
            case "camembert":
                self.sentence_encoder = SentenceTransformer(
                    "dangvantuan/sentence-camembert-large")
            case "french_semantic":
                self.sentence_encoder = SentenceTransformer(
                    "Sahajtomar/french_semantic")
            case "hugorosen_flaubert":
                self.sentence_encoder = SentenceTransformer(
                    "hugorosen/flaubert_base_uncased-xnli-sts")
            case "inokufu_flaubert":
                self.sentence_encoder = SentenceTransformer(
                    "inokufu/flaubert-base-uncased-xnli-sts")
            case "flaubert_education":
                self.sentence_encoder = SentenceTransformer(
                    "inokufu/flaubert-base-uncased-xnli-sts-finetuned-education")
            case _:
                raise NotImplementedError(
                    f"Sentence encoder: {sentence_encoder} is not implemented.")

    @deal.pre(lambda _: _.n_sentences > 0)
    def __call__(self,
                 doc: str,
                 n_sentences: int,
                 sent_pred: Callable[[str], object] = lambda _: True
                ) -> str:
        # TODO: `sentences` can be ran through a `thru` IterChain
        sentences = self.get_sentences(doc, sent_pred)
        embedded_sentences = self.get_sentence_encoding(sentences)
        ranks = self.textrank(embedded_sentences)
        top_sentences = self.select_top_k_texts_preserving_order(
            sentences, ranks, n_sentences)
        summary = '\n'.join(top_sentences)

        return summary

    @staticmethod
    def sentencizer(text: str) -> list[str]:
        return list(map(str, FrenchTextRank.nlp(text).sents))

    @staticmethod
    @deal.raises(TypeError, ValueError)
    def sim(u, v):
        return abs(1 - distance.cdist(u, v, 'cosine'))

    # Not used since using distance.cdist(u, v, 'cosine')
    # @staticmethod
    # @deal.pure
    # def cosine(u, v):
    #     return abs(1 - distance.cosine(u, v))


    # Call to this was commented out in original code
    # Decide necessity on revision
    # @staticmethod
    # @deal.pure
    # def rescale(a):

    #     maximum = np.max(a)
    #     minimum = np.min(a)

    #     return (a - minimum) / (maximum - minimum)

    @staticmethod
    @deal.has()
    def normalize(matrix):

        for row in matrix:
            row_sum = np.sum(row)
            if row_sum != 0:
                row /= row_sum

        return matrix

    @deal.raises(ValueError, TypeError, ValueError)
    @deal.pre(lambda _: 0.0 <= _.similarity_threshold <= 1.0)
    def textrank(self, texts_embeddings, similarity_threshold=0.8):

        matrix = FrenchTextRank.sim(texts_embeddings, texts_embeddings)
        np.fill_diagonal(matrix, 0)
        matrix[matrix < similarity_threshold] = 0

        matrix = FrenchTextRank.normalize(matrix)

        scaled_matrix = self.damping_factor * matrix
        scaled_matrix = FrenchTextRank.normalize(scaled_matrix)
        # scaled_matrix = rescale(scaled_matrix)

        ranks = np.ones((len(matrix), 1)) / len(matrix)
        iterations = 80
        for _ in range(iterations):
            ranks = scaled_matrix.T.dot(ranks)

        return ranks

    def get_sentence_encoding(self, text):

        if isinstance(text, (list, tuple)):
            return self.sentence_encoder.encode(text)

        return self.sentence_encoder.encode([text])

    @deal.pure
    @deal.pre(lambda _: _.k > 1)
    def select_top_k_texts_preserving_order(self, texts, ranking, k: int) -> list[str]:

        indexed_texts: FrenchTextRank.IndexedTexts
        top_ranked_texts: list[FrenchTextRank.RankedText]

        indexed_texts = enumerate(texts)
        top_ranked_texts = heapq.nlargest(k, zip(indexed_texts, ranking), key=itemgetter(1))

        top_texts: Iterator[FrenchTextRank.IndexedText]
        top_texts = (indexed_text for indexed_text, _ in top_ranked_texts)

        top_texts_in_preserved_order = [text for _, text in sorted(top_texts, key=itemgetter(0))]

        return top_texts_in_preserved_order

    def get_sentences(self, text: str, sent_pred: Callable[[str], object]) -> list[str]:

        paragraphs = text.split('\n')
        sentences = []
        for paragraph in paragraphs:
            sentences += self.sentencizer(paragraph)
        sentences = [s for s in sentences
                     if s and not s.isspace()
                        and sent_pred(s)]

        return sentences

if __name__ == "__main__":
    main()
