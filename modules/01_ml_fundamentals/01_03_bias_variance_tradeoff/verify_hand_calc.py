"""Verify fixed teaching calculations for Lesson 01.03."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def squared_losses(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have matching shapes")
    return (targets - predictions) ** 2


def mean_squared_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    losses = squared_losses(targets, predictions)
    if losses.size == 0:
        raise ValueError("cannot calculate MSE for an empty dataset")
    return float(np.mean(losses))


def main() -> None:
    y_d1 = np.array([2.5, 4.5, 8.5, 14.5], dtype=np.float64)
    y_d2 = np.array([3.75, 6.75, 10.75], dtype=np.float64)
    y_test = np.array([1.75, 2.3125, 4.8125], dtype=np.float64)
    predictions = {
        "constant": (
            np.full(4, 7.0, dtype=np.float64),
            np.full(3, 7.0, dtype=np.float64),
            np.full(3, 7.0, dtype=np.float64),
        ),
        "quadratic": (
            np.array([2.0, 4.0, 8.0, 14.0], dtype=np.float64),
            np.array([3.25, 6.25, 10.25], dtype=np.float64),
            np.array([1.75, 2.3125, 4.8125], dtype=np.float64),
        ),
        "degree5": (
            np.array([2.5, 4.5, 8.5, 14.5], dtype=np.float64),
            np.array([3.25, 6.25, 10.25], dtype=np.float64),
            np.array([3.25, 6.25, 10.25], dtype=np.float64),
        ),
    }
    datasets = ((y_d1, 0), (y_d2, 1), (y_test, 2))
    expected = {
        "constant": (21.25, 8.229166666666667, 18.106770833333332),
        "quadratic": (0.25, 0.25, 0.0),
        "degree5": (0.0, 0.25, 15.7734375),
    }

    for targets in (y_d1, y_d2, y_test):
        assert targets.dtype == np.float64
        assert targets.ndim == 1
    for name, model_predictions in predictions.items():
        for (targets, index), expected_mse in zip(datasets, expected[name], strict=True):
            losses = squared_losses(targets, model_predictions[index])
            assert np.all(losses >= 0.0)
            np.testing.assert_allclose(
                mean_squared_error(targets, model_predictions[index]),
                expected_mse,
                rtol=RTOL,
                atol=ATOL,
            )

    np.testing.assert_allclose(
        squared_losses(y_test, predictions["degree5"][2]),
        [2.25, 15.50390625, 29.56640625],
        rtol=RTOL,
        atol=ATOL,
    )
    print("PASS")


if __name__ == "__main__":
    main()
