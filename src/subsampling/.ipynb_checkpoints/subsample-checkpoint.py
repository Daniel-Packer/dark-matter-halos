from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
import numpy as np
from jax import random, numpy as jnp, jit, vmap
from ott.tools.k_means import k_means


def kmeans_downsample(point_cloud, downsample_size, n_trials, pbar=False):
    sampled_point_clouds = jnp.zeros([n_trials, downsample_size, 3])
    sampled_weights = jnp.zeros([n_trials, downsample_size])

    progress_bar = tqdm(range(n_trials)) if pbar else range(n_trials)
    for t in progress_bar:
        kmeans = KMeans(n_clusters=downsample_size, n_init="auto", random_state=t)

        kmeans.fit(point_cloud)
        _labels, counts = np.unique(kmeans.labels_, return_counts=True)

        weights = counts / counts.sum()
        kmeans.cluster_centers_

        sampled_point_clouds = sampled_point_clouds.at[t].set(kmeans.cluster_centers_)
        sampled_weights = sampled_weights.at[t].set(weights)

    return sampled_point_clouds, sampled_weights


def uniform_downsample(point_cloud, downsample_size, n_trials):
    sampled_weights = jnp.ones([n_trials, downsample_size]) / downsample_size

    rng = np.random.default_rng(seed=0)

    sampled_point_clouds = rng.choice(
        point_cloud, size=[n_trials, downsample_size], axis=0
    )

    return sampled_point_clouds, sampled_weights

def kmeans_downsample_points(points_list, downsample_size, n_trials, pbar=False, rng=None, outer_loop_size=-1, require_convergence=False, max_iterations=300):
    rng = random.PRNGKey(0) if rng is None else rng
    rngs = random.split(rng, num=n_trials)


    def cluster_trial(point_cloud, weights, rng):
        return k_means(point_cloud, downsample_size, weights=weights, rng=rng, max_iterations=max_iterations)
    
    trialed_cluster = jit(vmap(cluster_trial, [None, None, 0]))

    cloud_sizes = [p.shape[0] for p in points_list]
    max_cloud_size = max(cloud_sizes)
    padded_clouds = jnp.stack([jnp.concatenate([cloud, jnp.zeros([max_cloud_size - cloud.shape[0], *cloud.shape[1:]])]) for cloud in points_list], axis=0)
    padded_weights = jnp.stack([jnp.concatenate([jnp.ones([cloud_size]), jnp.zeros([max_cloud_size - cloud_size])]) for cloud_size in cloud_sizes], axis=0)

    if outer_loop_size == -1:
        clusterer = vmap(trialed_cluster, [0, 0, None])
        output = clusterer(padded_clouds, padded_weights, rngs)
        centroid, assignment = output.centroids, output.assignment
        if require_convergence:
            try:
                assert jnp.all(output.converged)
            except AssertionError:
                raise AssertionError(f"Convergence failed :( in {100 * jnp.mean(1 - group_output.converged):.1f}% of cases")
    else:
        group_clusterer = jit(vmap(trialed_cluster, [0, 0, None]))
        n = padded_clouds.shape[0]
        assignments = []
        centroids = []
        for i in range(n // outer_loop_size + 1):
            group_output = group_clusterer(
                padded_clouds[i * outer_loop_size : (i + 1) * outer_loop_size],
                padded_weights[i * outer_loop_size : (i + 1) * outer_loop_size],
                rngs)
            assignments.append(group_output.assignment)
            centroids.append(group_output.centroids)
            if require_convergence:
                try:
                    assert jnp.all(group_output.converged)
                except AssertionError:
                    raise AssertionError(f"Convergence failed :( in {100 * jnp.mean(1 - group_output.converged):.1f}% of cases")
        centroid = jnp.concatenate(centroids, 0)
        assignment = jnp.concatenate(assignments, 0)

    valid_assignments = (1 / padded_weights[:, None, :]) * assignment
    
    def get_label_freqs(valid_assignment):
        labels, freqs = jnp.unique(valid_assignment, return_counts=True, size=downsample_size+1)
        return freqs[:downsample_size] / jnp.sum(freqs[:downsample_size])
    
    sampled_weights = vmap(vmap(get_label_freqs))(valid_assignments)
    return centroid, sampled_weights
        

# Old implementation (with sklearn)
# def kmeans_downsample_points(points_list, downsample_size, n_trials, pbar=False):
#     dimension = points_list[0].shape[-1]
#     sampled_points = jnp.zeros([len(points_list), n_trials, downsample_size, dimension])
#     sampled_weights = jnp.zeros([len(points_list), n_trials, downsample_size])

#     progress_bar = (
#         tqdm(list(enumerate(points_list))) if pbar else enumerate(points_list)
#     )
#     for i, point_cloud in progress_bar:
#         centers, weights = kmeans_downsample(
#             points_list[i], downsample_size, n_trials, pbar=False
#         )

#         sampled_points = sampled_points.at[i].set(centers)
#         sampled_weights = sampled_weights.at[i].set(weights)

#     return sampled_points, sampled_weights


def fix_torus(pts):
    points = pts.copy()
    coords_to_fix = jnp.any(points < 1e3, axis=0)
    for coord_to_fix in np.arange(3)[coords_to_fix]:

        shift = 75_000
        split = shift / 2

        to_shift = points[:, coord_to_fix] > split

        points[to_shift, coord_to_fix] = points[to_shift, coord_to_fix] - shift
    return points


def preprocess_pointcloud(pts):
    points = pts.copy()
    points = fix_torus(points)
    return points
