from jax import numpy as jnp, typing, Array


# Taken from https://github.com/captain-pool/covhalo
def regression(
    train_targets: typing.ArrayLike, distances: typing.ArrayLike, k: int = 3
) -> Array:
    idxs = jnp.argsort(distances, axis=0)
    idxs = idxs[:k, :]
    dists = jnp.take_along_axis(distances, idxs, axis=0)
    inv_dists = 1.0 / (jnp.abs(dists) + 1e-8)
    normalized = inv_dists / inv_dists.sum(axis=0)
    estimated_labels = train_targets[idxs]
    prediction = (estimated_labels * normalized).sum(axis=0)
    return prediction


# Different version that takes in complete distance matrix:
def regression2(
    train_targets: typing.ArrayLike,
    distance_matrix: typing.ArrayLike,
    train_indices: typing.ArrayLike,
    test_indices: typing.ArrayLike,
    k: int = 3,
    weighted: bool = True,
) -> Array:
    distances_to_training_points = distance_matrix[test_indices][:, train_indices]
    nearest_pts = jnp.argsort(distances_to_training_points, axis=1)[:, :k]
    if weighted:
        dists = jnp.take_along_axis(distances_to_training_points, nearest_pts, axis=1)
        inv_dists = 1.0 / (jnp.abs(dists) + 1e-8)
        weights = inv_dists / inv_dists.sum(axis=1, keepdims=True)
    else:
        weights = jnp.ones([1, k]) / k

    estimated_labels = train_targets[nearest_pts]
    prediction = (estimated_labels * weights).sum(axis=1)
    return prediction
