import pickle
from pathlib import Path
from matplotlib import pyplot as plt
from jax import numpy as jnp, random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import sample, distributions as dist
import arviz as az


rng = random.PRNGKey(0)

G = 4.30219372e10 / 1e6
