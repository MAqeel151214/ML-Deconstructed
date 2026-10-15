"""Verify the fixed teaching calculations for lesson 00.01.

This script performs prediction with already-known weights and bias. It does not
fit a model or modify the teaching data.
"""

import numpy as np


X = np.array(
    [
        [5, 2, 8],
        [8, 3, 4],
        [6, 1, 12],
        [10, 4, 3],
        [4, 2, 15],
        [7, 3, 6],
    ],
    dtype=int,
)
w = np.array([12, 8, -3], dtype=int)
b = 20
expected_predictions = np.array([72, 128, 64, 163, 39, 110], dtype=int)


def predict_one_loop(features: np.ndarray, weights: np.ndarray, bias: int) -> int:
    """Return one prediction by explicitly summing feature contributions."""
    if features.ndim != 1 or weights.ndim != 1:
        raise ValueError("features and weights must each be one-dimensional")
    if features.shape != weights.shape:
        raise ValueError("features and weights must have matching shapes")

    weighted_sum = 0
    for feature, weight in zip(features, weights):
        weighted_sum += feature * weight

    return weighted_sum + bias


def predict_batch_loop(
    feature_matrix: np.ndarray, weights: np.ndarray, bias: int
) -> np.ndarray:
    """Return one loop-based prediction for each feature-matrix row."""
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional")
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if feature_matrix.shape[1] != weights.shape[0]:
        raise ValueError("each feature column needs one matching weight")

    return np.array(
        [predict_one_loop(features, weights, bias) for features in feature_matrix],
        dtype=int,
    )


def main() -> None:
    """Run deterministic shape and arithmetic checks, then print PASS."""
    x_a = X[0]
    predictions_loop = predict_batch_loop(X, w, b)
    predictions_vectorized = X @ w + b

    assert X.shape == (6, 3)
    assert w.shape == (3,)
    assert predictions_vectorized.shape == (6,)

    np.testing.assert_array_equal(x_a * w, np.array([60, 16, -24]))
    assert x_a @ w == 52
    assert predict_one_loop(x_a, w, b) == 72
    assert x_a @ w + b == 72

    np.testing.assert_array_equal(predictions_loop, expected_predictions)
    np.testing.assert_array_equal(predictions_vectorized, expected_predictions)
    np.testing.assert_array_equal(predictions_loop, predictions_vectorized)

    print("PASS")


if __name__ == "__main__":
    main()
