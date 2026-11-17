"""Independent verifier for Lesson 02.08 frozen arithmetic."""

from itertools import combinations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


RTOL = 0.0
ATOL = 1e-12

X = np.arange(1.0, 9.0, dtype=np.float64).reshape(8, 1)
y = np.array([2.2, 1.8, 5.0, 4.8, 8.2, 8.0, 5.0, 5.0], dtype=np.float64)


def rss(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values)


def best_split(rows: list[list[float]], targets: list[float],
               indices: list[int], min_samples_leaf: int) -> dict | None:
    parent_rss = rss([targets[index] for index in indices])
    best = None
    for feature_index in range(len(rows[0])):
        values = sorted({rows[index][feature_index] for index in indices})
        for lower, upper in zip(values, values[1:]):
            threshold = (lower + upper) / 2.0
            left = [index for index in indices
                    if rows[index][feature_index] <= threshold]
            right = [index for index in indices
                     if rows[index][feature_index] > threshold]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            gain = (parent_rss
                    - rss([targets[index] for index in left])
                    - rss([targets[index] for index in right]))
            if best is None or gain > best["gain"]:
                best = {
                    "feature": feature_index,
                    "threshold": threshold,
                    "gain": gain,
                    "left": left,
                    "right": right,
                }
    return best


def tree_predict(
    rows: list[list[float]],
    targets: list[float],
    query_rows: list[list[float]] | None = None,
) -> list[float]:
    def build(indices: list[int], depth: int) -> dict:
        values = [targets[index] for index in indices]
        node = {"prediction": sum(values) / len(values), "indices": indices}
        if depth == 2:
            return node
        split = best_split(rows, targets, indices, 2)
        if split is None or split["gain"] <= 0.0:
            return node
        node.update(
            feature=split["feature"],
            threshold=split["threshold"],
            left=build(split["left"], depth + 1),
            right=build(split["right"], depth + 1),
        )
        return node

    root = build(list(range(len(targets))), 0)
    predictions = []
    for row in rows if query_rows is None else query_rows:
        node = root
        while "left" in node:
            node = (
                node["left"]
                if row[node["feature"]] <= node["threshold"]
                else node["right"]
            )
        predictions.append(node["prediction"])
    return predictions


def partition_rss(cuts: tuple[int, ...]) -> float:
    boundaries = (0, *cuts, len(y))
    return sum(
        rss(y[boundaries[index]:boundaries[index + 1]].tolist())
        for index in range(len(boundaries) - 1)
    )


def main() -> None:
    rows = X.tolist()
    targets = y.tolist()
    split = best_split(rows, targets, list(range(8)), 2)
    assert split is not None
    assert split["feature"] == 0
    np.testing.assert_allclose(split["threshold"], 2.5, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(split["gain"], 24.0, rtol=RTOL, atol=ATOL)

    predictions = np.asarray(tree_predict(rows, targets), dtype=np.float64)
    np.testing.assert_allclose(
        predictions,
        [2.0, 2.0, 4.9, 4.9, 6.55, 6.55, 6.55, 6.55],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        y - predictions,
        [0.2, -0.2, 0.1, -0.1, 1.65, 1.45, -1.55, -1.55],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(np.sum((y - predictions) ** 2), 9.73,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        tree_predict(rows, targets, [[2.5], [3.7], [4.5], [7.2]]),
        [2.0, 4.9, 4.9, 6.55],
        rtol=RTOL,
        atol=ATOL,
    )

    all_partitions = []
    for n_cuts in range(1, 4):
        for cuts in combinations(range(2, 7), n_cuts):
            boundaries = (0, *cuts, 8)
            if all(boundaries[i + 1] - boundaries[i] >= 2
                   for i in range(len(boundaries) - 1)):
                all_partitions.append((cuts, partition_rss(cuts)))
    np.testing.assert_allclose(min(value for _, value in all_partitions), 0.12,
                               rtol=RTOL, atol=ATOL)

    model = DecisionTreeRegressor(
        criterion="squared_error", max_depth=2, min_samples_leaf=2
    ).fit(X, y)
    np.testing.assert_array_equal(model.tree_.feature, [0, -2, 0, -2, -2])
    np.testing.assert_allclose(
        model.tree_.threshold, [2.5, -2.0, 4.5, -2.0, -2.0],
        rtol=RTOL, atol=ATOL,
    )
    np.testing.assert_array_equal(model.tree_.n_node_samples, [8, 2, 6, 2, 4])
    np.testing.assert_allclose(model.predict(X), predictions,
                               rtol=RTOL, atol=ATOL)

    X2 = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],
          [4.0, 2.0], [4.0, 2.0], [4.0, 2.0], [4.0, 2.0]]
    y2 = [2.2, 2.0, 4.8, 5.2, 8.1, 7.9, 8.2, 7.8]
    secondary = best_split(X2, y2, list(range(8)), 2)
    assert secondary is not None
    assert secondary["feature"] == 1
    np.testing.assert_allclose(secondary["threshold"], 1.5,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        tree_predict(X2, y2),
        [2.1, 2.1, 5.0, 5.0, 8.0, 8.0, 8.0, 8.0],
        rtol=RTOL,
        atol=ATOL,
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
