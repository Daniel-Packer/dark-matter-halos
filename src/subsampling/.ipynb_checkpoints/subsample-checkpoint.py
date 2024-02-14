from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
import jax.numpy as jnp
import numpy as np

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

    sampled_point_clouds = rng.choice(point_cloud, size=[n_trials, downsample_size], axis=0)

    return sampled_point_clouds, sampled_weights
    



def kmeans_downsample_points(points_list, downsample_size, n_trials, pbar=False):
    sampled_points = jnp.zeros([len(points_list), n_trials, downsample_size, 3])
    sampled_weights = jnp.zeros([len(points_list), n_trials, downsample_size])

    progress_bar = tqdm(list(enumerate(points_list))) if pbar else enumerate(points_list)
    for i, point_cloud in progress_bar:
        centers, weights = kmeans_downsample(points_list[i], downsample_size, n_trials, pbar=False)
            
        sampled_points = sampled_points.at[i].set(centers)
        sampled_weights = sampled_weights.at[i].set(weights)
            

    return sampled_points, sampled_weights


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