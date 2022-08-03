from collections.abc import Iterator, Callable
from typing import ClassVar, Literal, TypeAlias
import heapq
from operator import itemgetter
import numpy as np
import numpy.typing as npt
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from spacy.lang.fr import French
import deal

# TODO: Typehint & contract this
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

    def __init__(self,
                 damping_factor=0.8,
                 sentence_encoder: SentenceEncoder | None = "camembert"):

        self.damping_factor = damping_factor

        match sentence_encoder:
            case "camembert" | None:
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
                 sent_pred: Callable[[str], object]=lambda _: True
                ) -> str:

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

    # pylint: disable=invalid-name
    def sim(self, u, v):
        return abs(1 - distance.cdist(u, v, 'cosine'))

    def cosine(self, u, v):
        return abs(1 - distance.cosine(u, v))

    def rescale(self, a):

        maximum = np.max(a)
        minimum = np.min(a)

        return (a - minimum) / (maximum - minimum)

    def normalize(self, matrix):

        for row in matrix:
            row_sum = np.sum(row)
            # if row_sum != 0:
            row /= row_sum

        return matrix

    @deal.pre(lambda _: 0.0 <= _.similarity_threshold <= 1.0)
    def textrank(self, texts_embeddings, similarity_threshold=0.8):

        matrix = self.sim(texts_embeddings, texts_embeddings)
        np.fill_diagonal(matrix, 0)
        matrix[matrix < similarity_threshold] = 0

        matrix = self.normalize(matrix)

        scaled_matrix = self.damping_factor * matrix
        scaled_matrix = self.normalize(scaled_matrix)
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
