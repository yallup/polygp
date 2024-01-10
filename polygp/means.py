"""Some common simple mean functions for time series analysis."""
import jax
import jax.numpy as jnp


def exponential(theta, x):
    return theta[0] * jnp.exp(theta[1] * (x + theta[3])) + theta[2]


# @jax.jit


def logarithm(theta, x):
    return theta[0] * jnp.log(jnp.abs(theta[1] * (x + theta[3]))) + theta[2]


# @jax.jit


def null(theta, x):
    return 0.0 * x


@jax.jit
def constant(theta, x):
    return theta[0]


@jax.jit
def linear(theta, x):
    return theta[0] * (x) + theta[1]


# @jax.jit
# def linear_shift(theta, x):
#     return theta[0] * (x+theta[2]) + theta[1]


@jax.jit
def power(theta, x):
    return theta[0] * (x + theta[3]) ** theta[1] + theta[2]


@jax.jit
def power_shift(theta, x):
    return theta[0] * (x) ** theta[1] + theta[2]
    # return (x) ** theta[0] + theta[1]
