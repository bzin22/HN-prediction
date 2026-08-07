"""Metrics against values computed by hand, including the tie cases that bite.

Every expectation here is arithmetic written out in a comment, not a number captured
from a run. A metric that silently changes definition is worse than one that is missing,
because the results table still looks fine.

Synthetic values only. No data file, no network.
"""

import numpy as np
import pytest

from hn_upvotes.training.metrics import (
    average_ranks,
    mae,
    precision_at_k,
    report,
    rmse,
    spearman,
)

# One rank inversion in the middle: the model puts the 3rd-largest and 2nd-largest the
# wrong way round and gets the two ends right.
PREDICTIONS = np.array([1.0, 2.0, 3.0, 4.0])
ACTUALS = np.array([1.0, 3.0, 2.0, 4.0])


def test_rmse_and_mae_on_a_hand_computed_fixture():
    # Errors are [0, -1, +1, 0]. RMSE = sqrt((0+1+1+0)/4) = sqrt(0.5) = 0.7071.
    assert rmse(PREDICTIONS, ACTUALS) == pytest.approx(np.sqrt(0.5))
    # MAE = (0 + 1 + 1 + 0) / 4 = 0.5.
    assert mae(PREDICTIONS, ACTUALS) == pytest.approx(0.5)


def test_spearman_on_a_hand_computed_fixture():
    # Predicted ranks [1,2,3,4], actual ranks [1,3,2,4], so the rank differences are
    # [0,-1,1,0]. Spearman = 1 - 6*sum(d^2) / (n(n^2-1)) = 1 - 12/(4*15) = 0.8.
    assert spearman(PREDICTIONS, ACTUALS) == pytest.approx(0.8)


def test_spearman_is_the_same_on_raw_score_and_on_log1p():
    """The claim the module makes: log1p is increasing, so it cannot change a rank."""
    scores = np.array([1.0, 2.0, 7.0, 40.0, 300.0, 3000.0])
    predictions = np.array([0.5, 3.0, 1.0, 9.0, 2.0, 4.0])
    assert spearman(predictions, scores) == pytest.approx(spearman(predictions, np.log1p(scores)))


def test_tied_values_share_their_average_rank():
    # Two fives occupy positions 1 and 2, so both rank 1.5, and the nine ranks 3.
    assert average_ranks(np.array([5.0, 5.0, 9.0])) == pytest.approx([1.5, 1.5, 3.0])


def test_spearman_of_a_constant_prediction_is_zero():
    """A model that predicts one number has no ranking, so it has no rank agreement."""
    assert spearman(np.full(5, 2.7), np.array([1.0, 5.0, 2.0, 9.0, 3.0])) == 0.0


def test_precision_at_k_on_a_hand_computed_fixture():
    # The model's top 2 are rows 3 and 2, holding actuals 4.0 and 2.0. The 2nd-largest
    # actual is 3.0, so only row 3 qualifies: 1 of 2 = 0.5.
    assert precision_at_k(PREDICTIONS, ACTUALS, k=2) == pytest.approx(0.5)


def test_precision_at_k_is_one_when_the_ranking_is_perfect():
    assert precision_at_k(PREDICTIONS, PREDICTIONS, k=2) == pytest.approx(1.0)


def test_a_row_level_with_the_kth_place_counts_as_being_in_the_top_k():
    """57.6% of stories score 1 or 2, so ties are the normal case, not an edge case."""
    # Every actual is 5, so the 2nd-largest is 5 and every pick is level with it.
    assert precision_at_k(PREDICTIONS, np.full(4, 5.0), k=2) == pytest.approx(1.0)


def test_asking_for_more_rows_than_exist_is_an_error():
    with pytest.raises(ValueError, match="cannot take the top 100"):
        precision_at_k(PREDICTIONS, ACTUALS, k=100)


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ValueError, match="differ in shape"):
        rmse(PREDICTIONS, ACTUALS[:2])


def test_report_carries_every_metric_and_the_row_count():
    result = report(PREDICTIONS, ACTUALS, k=2)
    assert result.rmse_log1p_score == pytest.approx(np.sqrt(0.5))
    assert result.mae_log1p_score == pytest.approx(0.5)
    assert result.spearman_raw_score == pytest.approx(0.8)
    assert result.precision_at_100 == pytest.approx(0.5)
    assert result.n_rows == 4
