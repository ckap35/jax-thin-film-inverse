import jax
import jax.numpy as jnp
import numpy as np

from thin_film_inverse import mean_squared_error, reflectance
from thin_film_inverse.experiment import (
    INDICES,
    WAVELENGTHS_NM,
    _fit,
    _objective,
    _scipy_objective,
)


def test_zero_thickness_is_bare_incident_substrate_interface():
    expected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
    actual = reflectance(0.0, jnp.array([400.0, 600.0, 800.0]), 1.0, 2.0, 1.5)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_film_matching_substrate_is_thickness_independent():
    expected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
    actual = reflectance(
        jnp.array([0.0, 73.0, 500.0]), 600.0, 1.0, 1.5, 1.5
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_matched_indices_have_zero_reflectance():
    actual = reflectance(jnp.array([0.0, 100.0, 600.0]), 600.0, 1.5, 1.5, 1.5)
    np.testing.assert_allclose(actual, 0.0, rtol=0.0, atol=1e-28)


def test_quarter_wave_antireflection_condition():
    actual = reflectance(100.0, 600.0, 1.0, 1.5, 2.25)
    np.testing.assert_allclose(actual, 0.0, rtol=0.0, atol=1e-28)


def test_automatic_derivative_matches_central_difference_away_from_optimum():
    measured = reflectance(180.0, WAVELENGTHS_NM, *INDICES)
    loss_grad = jax.value_and_grad(_objective)
    _, derivative = loss_grad(150.0, WAVELENGTHS_NM, measured)
    step_nm = 1e-3
    finite_difference = (
        mean_squared_error(150.0 + step_nm, WAVELENGTHS_NM, measured, *INDICES)
        - mean_squared_error(150.0 - step_nm, WAVELENGTHS_NM, measured, *INDICES)
    ) / (2.0 * step_nm)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-7, atol=1e-12)


def test_loop_vmap_and_jit_agree_for_candidate_losses():
    from thin_film_inverse.experiment import _loop_batch

    measured = reflectance(180.0, WAVELENGTHS_NM, *INDICES)
    candidates = jnp.linspace(0.0, 600.0, 601)
    vmap_objective = jax.vmap(_objective, in_axes=(0, None, None))
    eager = vmap_objective(candidates, WAVELENGTHS_NM, measured)
    compiled = jax.jit(vmap_objective)(candidates, WAVELENGTHS_NM, measured)
    loop = _loop_batch(candidates, WAVELENGTHS_NM, measured)
    assert eager.shape == (601,)
    np.testing.assert_allclose(loop, eager, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(compiled, eager, rtol=1e-13, atol=1e-15)


def test_noiseless_fit_recovers_thickness_from_suitable_basin():
    measured = reflectance(180.0, WAVELENGTHS_NM, *INDICES)
    result = _fit(WAVELENGTHS_NM, measured, 150.0)
    assert result["success"]
    np.testing.assert_allclose(result["thickness_nm"], 180.0, atol=1e-4)


def test_optimizer_gradient_accounts_for_scaled_parameter():
    measured = reflectance(180.0, WAVELENGTHS_NM, *INDICES)
    _, gradient_u = _scipy_objective(
        np.array([1.5]), WAVELENGTHS_NM, measured
    )
    derivative_nm = jax.grad(_objective)(150.0, WAVELENGTHS_NM, measured)
    np.testing.assert_allclose(gradient_u[0], 100.0 * derivative_nm, rtol=1e-13)
