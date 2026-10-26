"""Verify fixed teaching calculations for lesson 00.02.

The coordinates are synthetic teaching values in metres. This script validates
matrix operations and transformations; it does not fit or evaluate a model.
"""

import numpy as np


S = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=float)
R = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
I = np.eye(2, dtype=float)
P = np.array([[1.0, 2.0, -1.0], [2.0, -1.0, 0.0]], dtype=float)


def require_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    """Raise a clear error for a teaching array with the wrong shape."""
    if array.shape != shape:
        raise AssertionError(f"{name}.shape is {array.shape}; expected {shape}.")


def main() -> None:
    """Run deterministic shape, arithmetic, and convention checks."""
    p_a = P[:, :1]
    T = R @ S
    scaled_points = S @ P
    transformed_points = T @ P
    reverse_T = S @ R
    reverse_order_points = reverse_T @ P
    P_rows = P.T
    rowwise_transformed_points = P_rows @ T.T

    for name, array, shape in (
        ("S", S, (2, 2)),
        ("R", R, (2, 2)),
        ("I", I, (2, 2)),
        ("P", P, (2, 3)),
        ("p_a", p_a, (2, 1)),
        ("T", T, (2, 2)),
        ("transformed_points", transformed_points, (2, 3)),
        ("P_rows", P_rows, (3, 2)),
        ("rowwise_transformed_points", rowwise_transformed_points, (3, 2)),
    ):
        require_shape(name, array, shape)
        if array.ndim != 2:
            raise AssertionError(f"{name} must be two-dimensional.")

    np.testing.assert_array_equal(S + I, [[3.0, 0.0], [0.0, 1.5]])
    np.testing.assert_array_equal(0.5 * S, [[1.0, 0.0], [0.0, 0.25]])
    np.testing.assert_array_equal(S @ p_a, [[2.0], [1.0]])
    np.testing.assert_array_equal(
        scaled_points, [[2.0, 4.0, -2.0], [1.0, -0.5, 0.0]]
    )
    np.testing.assert_array_equal(T, [[0.0, -0.5], [2.0, 0.0]])
    np.testing.assert_array_equal(
        transformed_points, [[-1.0, 0.5, 0.0], [2.0, 4.0, -2.0]]
    )
    np.testing.assert_array_equal(R @ (S @ P), (R @ S) @ P)
    np.testing.assert_array_equal(reverse_T, [[0.0, -2.0], [0.5, 0.0]])
    np.testing.assert_array_equal(
        reverse_order_points, [[-4.0, 2.0, 0.0], [0.5, 1.0, -0.5]]
    )
    if np.array_equal(T, reverse_T):
        raise AssertionError("Expected R @ S and S @ R to differ.")
    np.testing.assert_array_equal(rowwise_transformed_points, transformed_points.T)

    print("PASS")


if __name__ == "__main__":
    main()
