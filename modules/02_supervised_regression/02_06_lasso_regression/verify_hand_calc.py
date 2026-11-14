"""Verify the corrected Lesson 02.06 Lasso fixture."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3] / "docs" / "lessons" / "02_06_lasso_regression"))
from implementation import lasso_coordinate_descent, soft_threshold


X = np.array([
    [0.4082482904638631, 0.5], [0.4082482904638631, -0.5],
    [0.4082482904638631, 0.0], [-0.4082482904638631, 0.0],
    [-0.4082482904638631, 0.5], [-0.4082482904638631, -0.5],
], dtype=np.float64)
y = np.array([0.712372, 0.512372, 1.887117, 1.787117, -1.757117, -1.917117], dtype=np.float64)


def main() -> None:
    assert X.shape == (6, 2) and y.shape == (6,)
    np.testing.assert_allclose(X.T @ X, np.eye(2), atol=1e-12, rtol=0)
    np.testing.assert_allclose(soft_threshold(-3.0, 1.0), -2.0, atol=1e-12, rtol=0)
    for lam, expected in {
        0.0: [0.204124, 2.040824220927, 0.18],
        0.05: [0.204124, 1.890824220927, 0.03],
        0.20: [0.204124, 1.440824220927, 0.0],
    }.items():
        theta = lasso_coordinate_descent(X, y, lam)
        np.testing.assert_allclose(theta, expected, atol=1e-8, rtol=0)
        predictions = theta[0] + X @ theta[1:]
        residuals = y - predictions
        rss = np.sum(residuals**2)
        mse = rss / y.size
        objective = mse + lam * np.sum(np.abs(theta[1:]))
        assert np.isfinite(objective)
    assert lasso_coordinate_descent(X, y, 0.20)[2] == 0.0
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
