"""Verify the fixed multiple-regression teaching calculations."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def main() -> None:
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    y = np.array([5.0, 1.0, 4.0, 8.0, 3.0], dtype=np.float64)
    X_new = np.array(
        [[3.0, 1.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float64
    )

    x_bar = np.mean(X, axis=0, dtype=np.float64)
    y_bar = np.float64(np.mean(y, dtype=np.float64))
    centered_X = X - x_bar
    centered_y = y - y_bar
    beta = np.linalg.solve(centered_X.T @ centered_X, centered_X.T @ centered_y)
    intercept = np.float64(y_bar - x_bar @ beta)
    fitted = np.asarray(intercept + X @ beta, dtype=np.float64)
    residuals = np.asarray(y - fitted, dtype=np.float64)
    rss = np.float64(np.sum(residuals**2, dtype=np.float64))
    mse = np.float64(rss / y.size)
    predictions = np.asarray(intercept + X_new @ beta, dtype=np.float64)

    np.testing.assert_allclose(
        x_bar, [1.0, 1.0], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        [y_bar, intercept, rss, mse],
        [4.2, 1.7, 0.3, 0.06],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        beta, [3.5, -1.0], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        fitted, [5.2, 0.7, 4.2, 7.7, 3.2], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        residuals, [-0.2, 0.3, -0.2, 0.3, -0.2],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        predictions, [11.2, -0.3, 6.7], rtol=RTOL, atol=ATOL
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
