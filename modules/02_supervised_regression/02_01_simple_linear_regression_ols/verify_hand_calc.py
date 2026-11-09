"""Verify the fixed OLS teaching calculations for Lesson 02.01."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def main() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([2.0, 3.0, 5.0, 4.0], dtype=np.float64)
    x_new = np.array([5.0, 6.0], dtype=np.float64)

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    slope = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    intercept = y_bar - slope * x_bar
    fitted = intercept + slope * x
    residuals = y - fitted
    rss = np.sum(residuals**2)
    mse = rss / x.size
    predictions = intercept + slope * x_new

    np.testing.assert_allclose(
        [x_bar, y_bar, slope, intercept, rss, mse],
        [2.5, 3.5, 0.8, 1.5, 1.8, 0.45],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        fitted, [2.3, 3.1, 3.9, 4.7], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        residuals, [-0.3, -0.1, 1.1, -0.7], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        predictions, [5.5, 6.3], rtol=RTOL, atol=ATOL
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
