import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize

from .model import mean_squared_error, reflectance

WAVELENGTHS_NM = jnp.linspace(400.0, 800.0, 201)
TRUE_THICKNESS_NM = 180.0
INDICES = (1.0, 2.0, 1.5)
THICKNESS_GRID_NM = jnp.linspace(0.0, 600.0, 601)
FINITE_DIFFERENCE_STEPS_NM = (10.0, 1.0, 0.1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11)
NOISE_SEED = 2026
NOISE_STANDARD_DEVIATION = 0.005
FIT_OPTIONS = {"gtol": 1e-10, "ftol": 1e-14, "maxiter": 200}


def _objective(thickness_nm, wavelength_nm, measured_reflectance):
    return mean_squared_error(
        thickness_nm, wavelength_nm, measured_reflectance, *INDICES
    )


def _scipy_objective(u, wavelength_nm, measured_reflectance):
    loss, derivative_nm = jax.value_and_grad(_objective)(
        100.0 * u[0], wavelength_nm, measured_reflectance
    )
    return float(loss), np.array([float(100.0 * derivative_nm)])


def _fit(wavelength_nm, measured_reflectance, start_nm):
    result = minimize(
        _scipy_objective,
        x0=np.array([start_nm / 100.0]),
        args=(wavelength_nm, measured_reflectance),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 6.0)],
        options=FIT_OPTIONS,
    )
    return {
        "start_nm": float(start_nm),
        "thickness_nm": float(100.0 * result.x[0]),
        "loss": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
    }


def _loop_batch(thicknesses_nm, wavelength_nm, measured_reflectance):
    return jnp.stack(
        [
            _objective(thickness_nm, wavelength_nm, measured_reflectance)
            for thickness_nm in thicknesses_nm
        ]
    )


def _measure(fn, args, repeats=7):
    start = perf_counter()
    first_result = fn(*args)
    first_result.block_until_ready()
    first_call_ms = (perf_counter() - start) * 1000.0

    fn(*args).block_until_ready()
    samples_ms = []
    for _ in range(repeats):
        start = perf_counter()
        fn(*args).block_until_ready()
        samples_ms.append((perf_counter() - start) * 1000.0)
    return {
        "first_call_ms": first_call_ms,
        "steady_state_median_ms": float(np.median(samples_ms)),
        "steady_state_samples_ms": samples_ms,
        "repeats": repeats,
    }


def _save_figures(output_dir, wavelength_nm, measured, loss_grid, fits, fd_rows, timings):
    wavelength_values = np.asarray(wavelength_nm)
    thickness_values = np.asarray(THICKNESS_GRID_NM)
    loss_values = np.asarray(loss_grid)

    fig, ax = plt.subplots()
    ax.plot(wavelength_values, np.asarray(measured), color="navy")
    ax.set(xlabel="Vacuum wavelength (nm)", ylabel="Reflectance",
           title="Synthetic reflectance spectrum (180 nm film)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_spectrum.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogy(thickness_values, np.maximum(loss_values, 1e-18))
    ax.axvline(TRUE_THICKNESS_NM, color="black", linestyle="--",
               label="Synthetic thickness")
    for fit in fits:
        ax.axvline(
            fit["thickness_nm"],
            linestyle=":",
            linewidth=1.2,
            label=(
                f"{fit['start_nm']:.0f} nm start -> "
                f"{fit['thickness_nm']:.2f} nm fit"
            ),
        )
    ax.set(xlabel="Film thickness (nm)", ylabel="Mean squared reflectance residual",
           title="Objective landscape and multistart fits")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "objective_landscape.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    steps = np.array([row["step_nm"] for row in fd_rows])
    errors = np.array([row["absolute_error"] for row in fd_rows])
    ax.loglog(steps, errors, marker="o", label="Absolute error")
    ax.set(xlabel="Central-difference step (nm)",
           ylabel="Absolute derivative error (per nm)",
           title="Automatic differentiation vs. central differences")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "derivative_check.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    labels = ["Python loop", "vmap", "vmap + jit"]
    values = [timings[label]["steady_state_median_ms"] for label in labels]
    ax.bar(labels, values, color=["slategray", "steelblue", "darkcyan"])
    ax.set(ylabel="Median time for 601 losses (ms)",
           title="Batch evaluation timings")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "batch_timings.png", dpi=160)
    plt.close(fig)


def run_experiment(output_dir=Path("results")):
    """Run the synthetic fit, derivative check, benchmarks, and save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wavelengths_nm = WAVELENGTHS_NM
    measured = reflectance(TRUE_THICKNESS_NM, wavelengths_nm, *INDICES)
    objective_grad = jax.value_and_grad(_objective)

    loss_150, derivative_150 = objective_grad(150.0, wavelengths_nm, measured)
    derivative_rows = []
    for step_nm in FINITE_DIFFERENCE_STEPS_NM:
        loss_plus = _objective(150.0 + step_nm, wavelengths_nm, measured)
        loss_minus = _objective(150.0 - step_nm, wavelengths_nm, measured)
        finite_difference = (loss_plus - loss_minus) / (2.0 * step_nm)
        derivative_rows.append({
            "step_nm": step_nm,
            "finite_difference_per_nm": float(finite_difference),
            "absolute_error": float(jnp.abs(finite_difference - derivative_150)),
        })

    fits = [_fit(wavelengths_nm, measured, start) for start in
            (50.0, 150.0, 350.0, 550.0)]
    loss_grid = jax.vmap(_objective, in_axes=(0, None, None))(
        THICKNESS_GRID_NM, wavelengths_nm, measured
    )

    noisy = measured + jax.random.normal(
        jax.random.PRNGKey(NOISE_SEED), measured.shape, dtype=measured.dtype
    ) * NOISE_STANDARD_DEVIATION
    noisy_fit = _fit(wavelengths_nm, noisy, 150.0)
    noisy_fit.update({
        "noise_seed": NOISE_SEED,
        "noise_standard_deviation_reflectance": NOISE_STANDARD_DEVIATION,
        "recovery_error_nm": noisy_fit["thickness_nm"] - TRUE_THICKNESS_NM,
    })

    batched_objective = jax.vmap(_objective, in_axes=(0, None, None))
    compiled_batch = jax.jit(batched_objective)
    benchmark_args = (THICKNESS_GRID_NM, wavelengths_nm, measured)
    batch_methods = {
        "Python loop": _loop_batch,
        "vmap": batched_objective,
        "vmap + jit": compiled_batch,
    }
    timings = {
        label: _measure(fn, benchmark_args)
        for label, fn in batch_methods.items()
    }
    loop_values = _loop_batch(*benchmark_args)
    vmap_values = batched_objective(*benchmark_args)
    compiled_values = compiled_batch(*benchmark_args)
    loop_values.block_until_ready()
    vmap_values.block_until_ready()
    compiled_values.block_until_ready()
    agreement = {
        "candidate_count": int(THICKNESS_GRID_NM.shape[0]),
        "output_shape": list(vmap_values.shape),
        "max_abs_loop_vmap": float(jnp.max(jnp.abs(loop_values - vmap_values))),
        "max_abs_vmap_jit": float(jnp.max(jnp.abs(vmap_values - compiled_values))),
    }

    _save_figures(
        output_dir, wavelengths_nm, measured, loss_grid, fits, derivative_rows,
        timings,
    )
    results = {
        "model": {
            "description": "Normal-incidence reflection from a finite coherent film",
            "assumptions": [
                "Real, wavelength-independent refractive indices",
                "Incident and substrate media are semi-infinite",
                "Thickness and vacuum wavelength are in nanometers",
            ],
            "indices_n0_n1_n2": list(INDICES),
            "wavelength_range_nm": [400.0, 800.0],
            "wavelength_count": int(wavelengths_nm.shape[0]),
            "true_thickness_nm": TRUE_THICKNESS_NM,
            "measurements": "synthetic",
            "precision": "float64",
        },
        "reference_point": {
            "thickness_nm": 150.0,
            "loss": float(loss_150),
            "derivative_per_nm": float(derivative_150),
        },
        "multistart_fits": fits,
        "noisy_recovery": noisy_fit,
        "derivative_check": {
            "evaluation_thickness_nm": 150.0,
            "automatic_derivative_per_nm": float(derivative_150),
            "steps": derivative_rows,
        },
        "batch_agreement": agreement,
        "timings": {
            "workload": "601 candidate thicknesses against one fixed 201-point spectrum",
            "precision": "float64",
            "synchronization": "block_until_ready",
            "plotting_included": False,
            "repeats": 7,
            "methods": timings,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main():
    results = run_experiment()
    print(json.dumps({
        "output_dir": str(Path("results")),
        "reference_point": results["reference_point"],
        "multistart_fits": results["multistart_fits"],
        "noisy_recovery": results["noisy_recovery"],
        "batch_agreement": results["batch_agreement"],
        "timings_ms": {
            label: {
                "first_call": measurements["first_call_ms"],
                "steady_median": measurements["steady_state_median_ms"],
            }
            for label, measurements in results["timings"]["methods"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
