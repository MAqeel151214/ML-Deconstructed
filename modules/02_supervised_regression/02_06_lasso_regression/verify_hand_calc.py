"""Verify the corrected Lesson 02.06 Lasso fixture."""

import numpy as np


def soft_threshold(z: float, gamma: float) -> np.float64:
    """Return the scalar soft-thresholding operator."""
    return np.float64(np.sign(z) * max(abs(z) - gamma, 0.0))


def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Fit an intercept-plus-features Lasso model with coordinate descent."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = np.float64(np.mean(y))
    for _ in range(max_iter):
        previous = np.r_[b, w].copy()
        for j in range(n_features):
            residual = y - b - X @ w + X[:, j] * w[j]
            rho = np.float64(X[:, j] @ residual)
            denominator = np.float64(X[:, j] @ X[:, j])
            if denominator == 0:
                raise ValueError("X cannot contain a zero-norm feature")
            w[j] = soft_threshold(rho, X.shape[0] * lam / 2.0) / denominator
        b = np.float64(np.mean(y - X @ w))
        if np.max(np.abs(np.r_[b, w] - previous)) < tol:
            break
    return np.asarray(np.r_[b, w], dtype=np.float64)


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
