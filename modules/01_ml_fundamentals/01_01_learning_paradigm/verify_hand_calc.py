"""Verify fixed teaching calculations for Lesson 01.01."""

import numpy as np


ATOL = 1e-12


def predict_linear(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + float(b)


def compute_residuals(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    return y - y_hat


def summarize_errors(residuals: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "mse": float(np.mean(residuals**2)),
    }


def main() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64)
    w1 = np.array([2.0], dtype=np.float64)
    w2 = np.array([1.5], dtype=np.float64)
    y_hat_1 = predict_linear(X, w1, 1.0)
    y_hat_2 = predict_linear(X, w2, 1.0)
    e_1 = compute_residuals(y, y_hat_1)
    e_2 = compute_residuals(y, y_hat_2)
    s1 = summarize_errors(e_1)
    s2 = summarize_errors(e_2)

    assert X.shape == (4, 1) and y.shape == (4,)
    assert X.dtype == np.float64 and y.dtype == np.float64
    np.testing.assert_allclose(y_hat_1, [3, 5, 7, 9], rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(e_1, [0, 0, 0, 0], rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(y_hat_2, [2.5, 4, 5.5, 7], rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(e_2, [0.5, 1, 1.5, 2], rtol=0.0, atol=ATOL)
    assert e_2[0] == 0.5 and e_2[-1] == 2.0
    assert s1 == {"mae": 0.0, "mse": 0.0}
    assert s2 == {"mae": 1.25, "mse": 1.875}

    for w, expected in ((w1, y_hat_1), (w2, y_hat_2)):
        predictions = []
        residuals = []
        abs_total = 0.0
        square_total = 0.0
        for idx in range(4):
            prediction = float(w[0]) * float(X[idx, 0]) + 1.0
            residual = float(y[idx]) - prediction
            predictions.append(prediction)
            residuals.append(residual)
            abs_total += abs(residual)
            square_total += residual**2
        np.testing.assert_allclose(predictions, expected, rtol=0.0, atol=ATOL)
        assert np.isclose(abs_total / 4, float(summarize_errors(np.array(residuals))["mae"]), atol=ATOL)
        assert np.isclose(square_total / 4, float(summarize_errors(np.array(residuals))["mse"]), atol=ATOL)

    X_aug = np.column_stack([X, np.ones(4, dtype=np.float64)])
    for w, expected in ((w1, y_hat_1), (w2, y_hat_2)):
        np.testing.assert_allclose(X_aug @ np.array([w[0], 1.0]), expected, rtol=0.0, atol=ATOL)
    print("PASS")


if __name__ == "__main__":
    main()
