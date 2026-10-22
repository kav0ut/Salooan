import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_pinball_loss

from src.metrics import breach_rate, crossing_rate, pinball_loss


def test_pinball_loss_hand_computed_symmetric() -> None:
    y = [1.0, 2.0, 3.0]
    q = [2.0, 2.0, 2.0]
    alpha = 0.5
    loss = pinball_loss(y, q, alpha)
    assert loss == pytest.approx(1.0 / 3.0)


def test_pinball_loss_asymmetry() -> None:
    alpha = 0.05
    # When y - q = +1: loss = alpha * 1 = 0.05
    loss_underestimate = pinball_loss([2.0], [1.0], alpha)
    assert loss_underestimate == pytest.approx(0.05)

    # When y - q = -1: loss = (alpha - 1) * (-1) = 0.95
    loss_overestimate = pinball_loss([1.0], [2.0], alpha)
    assert loss_overestimate == pytest.approx(0.95)


def test_pinball_loss_at_median_equals_half_mae() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    q = rng.normal(0.0, 1.0, size=200)

    loss_median = pinball_loss(y, q, alpha=0.5)
    mae = float(np.mean(np.abs(y - q)))
    assert loss_median == pytest.approx(0.5 * mae)


def test_pinball_loss_perfect_prediction_is_zero_and_non_negative() -> None:
    y = np.array([0.05, -0.12, 0.33, -0.04])
    # Perfect predictions give zero loss
    assert pinball_loss(y, y, alpha=0.05) == pytest.approx(0.0)
    assert pinball_loss(y, y, alpha=0.95) == pytest.approx(0.0)

    # Imperfect predictions must always be strictly non-negative
    rng = np.random.default_rng(99)
    y_rand = rng.uniform(-1.0, 1.0, size=100)
    q_rand = rng.uniform(-1.0, 1.0, size=100)
    for a in [0.01, 0.05, 0.5, 0.95, 0.99]:
        assert pinball_loss(y_rand, q_rand, alpha=a) >= 0.0


def test_pinball_loss_agrees_with_sklearn() -> None:
    rng = np.random.default_rng(123)
    y = rng.normal(0.0, 0.2, size=500)
    q = rng.normal(-0.1, 0.2, size=500)

    alphas = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for alpha in alphas:
        hand_loss = pinball_loss(y, q, alpha)
        sk_loss = mean_pinball_loss(y, q, alpha=alpha)
        assert hand_loss == pytest.approx(sk_loss, rel=1e-6)


def test_pinball_loss_minimizer_property() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=20000)
    alpha = 0.05

    # True theoretical minimizer is the alpha-quantile of the sample
    optimal_q = float(np.quantile(y, alpha))
    optimal_preds = np.full(len(y), optimal_q)
    optimal_loss = pinball_loss(y, optimal_preds, alpha)

    # Shifted predictions must have strictly higher pinball loss
    for shift in [-0.2, -0.1, 0.1, 0.2]:
        shifted_preds = np.full(len(y), optimal_q + shift)
        shifted_loss = pinball_loss(y, shifted_preds, alpha)
        assert shifted_loss > optimal_loss


def test_breach_rate_strict_inequality_and_extremes() -> None:
    # Strict inequality: equality is not a breach
    y_equal = [1.0, 2.0, 3.0]
    q_equal = [1.0, 2.0, 3.0]
    assert breach_rate(y_equal, q_equal) == pytest.approx(0.0)

    # Hand-computed: y < q on 1 out of 3 elements
    y = [1.0, 2.0, 3.0]
    q = [2.0, 2.0, 2.0]
    assert breach_rate(y, q) == pytest.approx(1.0 / 3.0)

    # Extreme: zero breaches
    assert breach_rate([1.0, 2.0], [0.0, 0.0]) == pytest.approx(0.0)

    # Extreme: 100% breaches
    assert breach_rate([1.0, 2.0], [5.0, 5.0]) == pytest.approx(1.0)


def test_validation_errors() -> None:
    # Empty inputs
    with pytest.raises(ValueError, match="empty"):
        pinball_loss([], [], alpha=0.05)
    with pytest.raises(ValueError, match="empty"):
        breach_rate([], [])

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        pinball_loss([1.0, 2.0], [1.0], alpha=0.05)
    with pytest.raises(ValueError, match="Length mismatch"):
        breach_rate([1.0, 2.0], [1.0])

    # NaN in inputs
    with pytest.raises(ValueError, match="NaN"):
        pinball_loss([np.nan, 1.0], [1.0, 1.0], alpha=0.05)
    with pytest.raises(ValueError, match="NaN"):
        pinball_loss([1.0, 1.0], [1.0, np.nan], alpha=0.05)
    with pytest.raises(ValueError, match="NaN"):
        breach_rate([np.nan, 1.0], [1.0, 1.0])

    # Alpha out of bounds (strict: 0 < alpha < 1)
    for bad_alpha in [0.0, 1.0, -0.05, 1.05]:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            pinball_loss([1.0, 2.0], [1.0, 2.0], alpha=bad_alpha)


def test_inputs_are_not_mutated() -> None:
    y_arr = np.array([1.0, 2.0, 3.0])
    q_arr = np.array([2.0, 2.0, 2.0])
    y_copy = y_arr.copy()
    q_copy = q_arr.copy()

    _ = pinball_loss(y_arr, q_arr, alpha=0.05)
    _ = breach_rate(y_arr, q_arr)

    np.testing.assert_array_equal(y_arr, y_copy)
    np.testing.assert_array_equal(q_arr, q_copy)

    y_series = pd.Series([1.0, 2.0, 3.0])
    q_series = pd.Series([2.0, 2.0, 2.0])
    y_ser_copy = y_series.copy()
    q_ser_copy = q_series.copy()

    _ = pinball_loss(y_series, q_series, alpha=0.05)
    _ = breach_rate(y_series, q_series)

    pd.testing.assert_series_equal(y_series, y_ser_copy)
    pd.testing.assert_series_equal(q_series, q_ser_copy)


def test_crossing_rate_monotone_gives_zero() -> None:
    # Monotone non-increasing returns as confidence increases yields 0 crossing
    preds = {
        0.90: np.array([-0.05, -0.04, -0.06]),
        0.95: np.array([-0.10, -0.08, -0.12]),
        0.99: np.array([-0.20, -0.18, -0.25]),
    }
    assert crossing_rate(preds) == pytest.approx(0.0)


def test_crossing_rate_one_violating_row_out_of_n() -> None:
    # 4 rows: row 1 violates (-0.05 < -0.02), other 3 rows are monotone
    preds = {
        0.90: [-0.05, -0.05, -0.05, -0.05],
        0.95: [-0.10, -0.02, -0.10, -0.10],
    }
    assert crossing_rate(preds) == pytest.approx(1.0 / 4.0)


def test_crossing_rate_ties_are_not_violations() -> None:
    # Exact ties across confidence levels are not violations
    preds = {
        0.90: [-0.10, -0.05, -0.02],
        0.95: [-0.10, -0.05, -0.02],
        0.99: [-0.10, -0.05, -0.02],
    }
    assert crossing_rate(preds) == pytest.approx(0.0)


def test_crossing_rate_three_confidences_only_last_pair_crosses_counted_once() -> None:
    # Row 0: 0.90 -> -0.05, 0.95 -> -0.10 (ok), 0.99 -> -0.08 (crosses 0.95)
    # Row 1: all ok (-0.05 >= -0.10 >= -0.20)
    preds = {
        0.90: [-0.05, -0.05],
        0.95: [-0.10, -0.10],
        0.99: [-0.08, -0.20],
    }
    # 1 of 2 rows violates, counted once per row -> 0.5
    assert crossing_rate(preds) == pytest.approx(0.5)


def test_crossing_rate_validation_errors() -> None:
    # Single confidence level
    with pytest.raises(ValueError, match="at least two"):
        crossing_rate({0.95: [-0.10, -0.05]})

    # Unequal lengths
    with pytest.raises(ValueError, match="Length mismatch"):
        crossing_rate({0.90: [-0.05, -0.05], 0.95: [-0.10]})

    # NaN in predictions
    with pytest.raises(ValueError, match="NaN"):
        crossing_rate({0.90: [-0.05, np.nan], 0.95: [-0.10, -0.10]})


def test_crossing_rate_inputs_not_mutated() -> None:
    arr1 = np.array([-0.05, -0.04])
    arr2 = np.array([-0.10, -0.08])
    copy1 = arr1.copy()
    copy2 = arr2.copy()

    _ = crossing_rate({0.90: arr1, 0.95: arr2})

    np.testing.assert_array_equal(arr1, copy1)
    np.testing.assert_array_equal(arr2, copy2)

