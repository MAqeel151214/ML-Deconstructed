"""Verify fixed teaching calculations for lesson 00.08."""

import numpy as np


def main() -> None:
    X = np.array([[1, 2], [3, 0], [2, 4], [5, 1]], dtype=np.int64)
    w = np.array([10, -2], dtype=np.int64)
    b = np.int64(3)
    expected = np.array([9, 33, 15, 51], dtype=np.int64)

    loop = np.empty(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        loop[i] = X[i, 0] * w[0] + X[i, 1] * w[1] + b

    contributions = X * w
    reduced = contributions.sum(axis=1) + b
    vectorized = X @ w + b

    cross = np.empty(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        total = b
        for j in range(X.shape[1]):
            total += X[i, j] * w[j]
        cross[i] = total

    assert X.shape == (4, 2)
    assert w.shape == (2,)
    assert b.shape == ()
    assert contributions.shape == (4, 2)
    assert vectorized.shape == (4,)
    assert (X @ w.reshape(2, 1)).shape == (4, 1)
    np.testing.assert_array_equal(loop, expected)
    np.testing.assert_array_equal(reduced, expected)
    np.testing.assert_array_equal(vectorized, expected)
    np.testing.assert_array_equal(cross, expected)

    for invalid_weights in (
        w.reshape(2, 1),
        np.array([1, 2, 3], dtype=np.int64),
    ):
        try:
            X * invalid_weights
        except ValueError:
            pass
        else:
            raise AssertionError("Expected incompatible elementwise shapes")

    print("PASS")


if __name__ == "__main__":
    main()
