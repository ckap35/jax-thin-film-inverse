import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def reflectance(thickness_nm, wavelength_nm, n0, n1, n2):
    """Return normal-incidence reflectance for a coherent finite film."""
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    delta = 2.0 * jnp.pi * n1 * thickness_nm / wavelength_nm
    q = jnp.exp(2j * delta)
    amplitude = (r01 + r12 * q) / (1.0 + r01 * r12 * q)
    return jnp.abs(amplitude) ** 2


def mean_squared_error(
    thickness_nm, wavelength_nm, measured_reflectance, n0, n1, n2
):
    """Mean squared reflectance residual for a fixed measured spectrum."""
    residual = (
        reflectance(thickness_nm, wavelength_nm, n0, n1, n2)
        - measured_reflectance
    )
    return jnp.mean(residual**2)
