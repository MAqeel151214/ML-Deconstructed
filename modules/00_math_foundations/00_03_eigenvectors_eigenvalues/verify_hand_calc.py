"""Verify fixed teaching calculations for lesson 00.03.

The matrices and vectors are dimensionless teaching values. This script verifies
known eigenpair candidates; it does not find eigenpairs numerically.
"""

from __future__ import annotations

import numpy as np


A = np.array([[2, 1], [1, 2]], dtype=int)
V_PLUS = np.array([1, 1], dtype=int)
V_MINUS = np.array([1, -1], dtype=int)
U = np.array([2, 1], dtype=int)
B = np.array([[-2, 0], [0, 1]], dtype=int)
W = np.array([1, 0], dtype=int)
ZERO = np.array([0, 0], dtype=int)


def require_square_matrix(name: str, matrix: np.ndarray) -> None:
    """Raise a clear error unless matrix has shape (n, n)."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AssertionError(f"{name} must have square shape (n, n); got {matrix.shape}.")


def require_compatible_vector(name: str, matrix: np.ndarray, vector: np.ndarray) -> None:
    """Raise a clear error unless vector has shape (n,) for matrix."""
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[0]:
        raise AssertionError(
            f"{name} must have shape ({matrix.shape[0]},); got {vector.shape}."
        )


def is_exact_eigenpair(matrix: np.ndarray, vector: np.ndarray, eigenvalue: int) -> bool:
    """Test A @ v == lambda * v after enforcing the non-zero-vector rule."""
    require_square_matrix("matrix", matrix)
    require_compatible_vector("vector", matrix, vector)
    if not np.any(vector != 0):
        return False
    return bool(np.array_equal(matrix @ vector, eigenvalue * vector))


def scalar_row_dot_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Independently compute a 2-D matrix-vector product one row at a time."""
    if matrix.shape != (2, 2) or vector.shape != (2,):
        raise AssertionError("This cross-check expects a (2, 2) matrix and a (2,) vector.")
    return np.array(
        [
            matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1],
            matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1],
        ],
        dtype=int,
    )


def main() -> None:
    """Run deterministic shape, arithmetic, eigenpair, and cross-check tests."""
    for name, matrix in (("A", A), ("B", B)):
        require_square_matrix(name, matrix)

    for name, matrix, vector in (
        ("V_PLUS", A, V_PLUS),
        ("V_MINUS", A, V_MINUS),
        ("U", A, U),
        ("W", B, W),
        ("ZERO", A, ZERO),
    ):
        require_compatible_vector(name, matrix, vector)

    np.testing.assert_array_equal(A @ V_PLUS, [3, 3])
    np.testing.assert_array_equal(A @ V_MINUS, [1, -1])
    np.testing.assert_array_equal(A @ U, [5, 4])
    np.testing.assert_array_equal(B @ W, [-2, 0])

    assert is_exact_eigenpair(A, V_PLUS, 3)
    assert is_exact_eigenpair(A, V_MINUS, 1)
    assert is_exact_eigenpair(B, W, -2)
    assert not is_exact_eigenpair(A, ZERO, 0)

    transformed_u = A @ U
    assert U[0] * transformed_u[1] - U[1] * transformed_u[0] == 3

    for matrix, vector in ((A, V_PLUS), (A, V_MINUS), (A, U), (B, W)):
        np.testing.assert_array_equal(
            matrix @ vector, scalar_row_dot_product(matrix, vector)
        )

    print("PASS")


if __name__ == "__main__":
    main()
