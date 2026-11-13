"""Verify the fixed Ridge Regression teaching calculations."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def main() -> None:
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    y = np.array([5.0, 1.0, 4.0, 8.0, 3.0], dtype=np.float64)
    A = np.column_stack((np.ones(5, dtype=np.float64), X))
    identity = np.diag([0.0, 1.0, 1.0]).astype(np.float64)
    expected = {
        0.0: np.array([1.7, 3.5, -1.0], dtype=np.float64),
        0.5: np.array([2.2, 2.8, -0.8], dtype=np.float64),
        5.0: np.array(
            [3.4857142857142858, 1.0, -0.2857142857142857],
            dtype=np.float64,
        ),
    }
    expected_rss = {0.0: 0.3, 0.5: 1.36, 5.0: 13.820408163265306}
    expected_penalty = {0.0: 0.0, 0.5: 4.24, 5.0: 5.408163265306122}
    expected_objective = {
        0.0: 0.3,
        0.5: 5.6,
        5.0: 19.228571428571428,
    }

    for lam, expected_theta in expected.items():
        theta = np.asarray(
            np.linalg.solve(
                A.T @ A + lam * identity,
                A.T @ y,
            ),
            dtype=np.float64,
        )
        predictions = np.asarray(A @ theta, dtype=np.float64)
        residuals = np.asarray(y - predictions, dtype=np.float64)
        rss = np.float64(np.sum(residuals**2, dtype=np.float64))
        penalty = np.float64(lam * np.sum(theta[1:] ** 2, dtype=np.float64))
        objective = np.float64(rss + penalty)

        np.testing.assert_allclose(theta, expected_theta, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            rss, expected_rss[lam], rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            penalty, expected_penalty[lam], rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            objective, expected_objective[lam], rtol=RTOL, atol=ATOL
        )

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
