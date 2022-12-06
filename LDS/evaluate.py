from collections import defaultdict, namedtuple
from typing import NamedTuple

import deal
from funcy import walk_values
import numpy as np
from rouge_score import rouge_scorer

try:
    import nltk
    # Needed for google's rouge_scorer's sent_tokenize
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt') # pyright: ignore[reportUnboundVariable]

@deal.pre(lambda _: len(_.predictions) == len(_.targets))
def evaluate(predictions: list[str],
             targets: list[str]
) -> dict[str, dict[str, float]]:

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        use_stemmer=False,
        split_summaries=True
    )
    all_rouge_scores = map(
        scorer.score,
        targets,
        predictions
    )
    aggregator = BootstrapAggregator()
    for score in all_rouge_scores:
        aggregator.add_scores(score)

    aggregated_scores = aggregator.aggregate()

    mean_scores = walk_values(
        lambda agg_score: walk_values(lambda mean: round(mean * 100, 2),
                                      agg_score.mid._asdict()),
        aggregated_scores)

    return mean_scores

# Re-exported from:
# https://github.com/google-research/google-research/blob/master/rouge/scoring.py
# Because they provide aggregate scoring with CLI file inputs but not through the API
class AggregateScore(namedtuple("AggregateScore", ["low", "mid", "high"])):
    """Tuple containing confidence intervals for scores."""

class BootstrapAggregator:

    @deal.pre(lambda _: 0 < _.confidence_interval < 1)
    @deal.pre(lambda _: _.n_samples > 0)
    def __init__(self, confidence_interval=0.95, n_samples=1000):

        self._n_samples = n_samples
        self._confidence_interval = confidence_interval
        self._scores = defaultdict(list)

    def add_scores(self, scores: dict[str, NamedTuple]) -> None:
        for score_type, score in scores.items():
            self._scores[score_type].append(score)

    def aggregate(self) -> dict[str, AggregateScore]:
        result = {}
        for score_type, scores in self._scores.items():
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple(
                (scores[0].__class__(*percentiles[j, :]) for j in range(3)))
            result[score_type] = AggregateScore(
                low=intervals[0], mid=intervals[1], high=intervals[2])

        return result

    @deal.has('random')
    def _bootstrap_resample(self, matrix):
        # Matrix of (bootstrap sample, measure).
        sample_mean = np.zeros((self._n_samples, matrix.shape[1]))
        for i in range(self._n_samples):
            sample_idx = np.random.choice(
                np.arange(matrix.shape[0]), size=matrix.shape[0])
            sample = matrix[sample_idx, :]
            sample_mean[i, :] = np.mean(sample, axis=0)

        # Take percentiles on the estimate of the mean using bootstrap samples.
        # Final result is a (bounds, measure) matrix.
        percentile_delta = (1 - self._confidence_interval) / 2
        q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])

        return np.percentile(sample_mean, q, axis=0)


@deal.has()
@deal.pre(lambda _: _.precision >= 0)
@deal.pre(lambda _: _.recall >= 0)
def fmeasure(precision: float, recall: float) -> float:

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)

    return 0.0
