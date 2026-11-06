"""Verify fixed teaching calculations for Lesson 01.04."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def mean_squared_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.dtype != np.float64 or predictions.dtype != np.float64:
        raise ValueError("arrays must use float64")
    if targets.shape != predictions.shape:
        raise ValueError("arrays must have matching shapes")
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("arrays must be non-empty 1-D arrays")
    return float(np.mean((targets - predictions) ** 2))


def main() -> None:
    targets = {
        "train": np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64),
        "validation": np.array([11.0, 13.0, 15.0], dtype=np.float64),
        "test": np.array([17.0, 19.0, 21.0, 23.0, 25.0], dtype=np.float64),
    }
    predictions = {
        "A": {
            "train": [3.0, 5.0, 7.0, 9.0],
            "validation": [11.0, 13.0, 15.0],
            "test": [17.0, 19.0, 21.0, 23.0, 25.0],
        },
        "B": {
            "train": [3.0, 5.0, 7.0, 9.0],
            "validation": [10.0, 12.5, 15.5],
            "test": [16.0, 18.5, 21.5, 24.5, 28.0],
        },
    }
    expected = {"A": (0.0, 0.0, 0.0), "B": (0.0, 0.5, 2.55)}
    splits = ("train", "validation", "test")
    for candidate, candidate_predictions in predictions.items():
        for split, expected_mse in zip(splits, expected[candidate], strict=True):
            prediction_array = np.array(candidate_predictions[split], dtype=np.float64)
            np.testing.assert_allclose(
                mean_squared_error(targets[split], prediction_array),
                expected_mse,
                rtol=RTOL,
                atol=ATOL,
            )

    validation_mse = {candidate: expected[candidate][1] for candidate in predictions}
    assert min(validation_mse, key=validation_mse.get) == "A"
    np.testing.assert_allclose(
        (1.0 + 0.25 + 0.25 + 2.25 + 9.0) / 5,
        2.55,
        rtol=RTOL,
        atol=ATOL,
    )
    print("PASS")


if __name__ == "__main__":
    main()
