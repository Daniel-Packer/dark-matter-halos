import pickle
from pathlib import Path
from matplotlib import pyplot as plt
from jax import numpy as jnp, random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import sample, distributions as dist
import arviz as az
import numpy as np


rng = random.PRNGKey(0)

G = 4.30219372e10 / 1e6

def load_data(data_path: Path) -> dict[str, jnp.ndarray]:
    halos_path = data_path / "halos.pkl"
    with open(halos_path, "rb") as f:
        particles, halo_slice  = pickle.load(f)

    train_indices = np.loadtxt(data_path / f"train_indices.txt").astype(int)
    val_indices = np.loadtxt(data_path / f"val_indices.txt").astype(int)
    test_indices = np.loadtxt(data_path / f"test_indices.txt").astype(int)

    labels = jnp.log10(
        halo_slice["GroupStellarMass"] * 1e10 / 0.677
    )  # label by logMstar
    logc = jnp.log10(
        halo_slice["SubhaloVmax"]
        / jnp.sqrt(G * halo_slice["Group_M_Mean200"] / halo_slice["Group_R_Mean200"])
    )
    logm = jnp.log10(jnp.array([particle["count"] for particle in particles]))


    # An arbitrary choice of scaling is needed here for the relative importance of the two variables
    # When I performed a linear regression (see `classical_approach_linear_model.ipynb`), I found that
    # logc was about 5 times as important as logm. (In fact, I did not even have strong evidence that
    # logm was important, since the Bayesian regression posterior distribution included zero).
    alpha = 5
    predictors = jnp.stack([logc * alpha, logm], axis=1)


    train_labels, val_labels, test_labels = labels[train_indices], labels[val_indices], labels[test_indices]
    train_predictors, val_predictors, test_predictors = predictors[train_indices], predictors[val_indices], predictors[test_indices]

    return {
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "train_predictors": train_predictors,
        "val_predictors": val_predictors,
        "test_predictors": test_predictors,
    }


def linear_regression(x_data, y_data=None):
    std = sample("std", dist.Exponential(1))
    intercept = sample("intercept", dist.Normal(0, 3))
    slopes = sample("slopes", dist.Normal(0, 3), sample_shape=(x_data.shape[-1],))
    
    mean_predictions = intercept + (x_data @ slopes)

    return sample("predictions", dist.Normal(mean_predictions, std), obs=y_data)