"""Verify the fixed polynomial-regression teaching calculations."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def main() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([4.1, 1.1, 1.6, 2.1, 6.1], dtype=np.float64)
    A = np.column_stack((np.ones(5, dtype=np.float64), x, x**2))
    theta = np.asarray(np.linalg.solve(A.T @ A, A.T @ y), dtype=np.float64)
    predictions = np.asarray(A @ theta, dtype=np.float64)
    residuals = np.asarray(y - predictions, dtype=np.float64)

    np.testing.assert_allclose(
        theta, [1.0, 0.5, 1.0], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        predictions, [4.0, 1.5, 1.0, 2.5, 6.0], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        residuals, [0.1, -0.4, 0.6, -0.4, 0.1], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.sum(residuals**2, dtype=np.float64), 0.7, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.mean(residuals**2, dtype=np.float64), 0.14, rtol=RTOL, atol=ATOL
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
