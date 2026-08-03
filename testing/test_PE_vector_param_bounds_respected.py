"""Regression tests: bounds on vector ``_p_est`` slots must be enforced.

setting scalar bounds on a
vector ``_p_est`` slot under a non-unit scaling produced an estimate four
orders of magnitude beyond the upper bound (``K_i = 1e5`` with
``upper=10.0`` and ``scaling=0.5``). A sibling vector parameter with the
same scalar-broadcast bound pattern but a much larger scaling was
respected, suggesting a scaling-dependent path that lets some vector
slots escape their bounds.

These tests assert the *contract* on the toy: a scalar bound assigned to
a vector slot, with non-unit scaling, must bind every element. The
contract is asserted for both the collocation and single-shooting backends
and for bounds set both before and after ``setup``.

Note on reproduction: on this small (3-element decoupled decay) model the
bug as originally reported has *not* been reproduced -- the optimizer is
correctly clamped at the upper bound under all tested combinations. The
tests below are guardrails for the contract; a faithful reproducer of the
108-state PID-gain pathology would need to be added separately by the
original regression report. If the contract regresses on this toy at any
point, these tests will catch it immediately.
"""

from __future__ import annotations
import unittest

import numpy as np
import pandas as pd
import pytest
import do_mpc

from do_mpc.estimator import ParameterEstimator, Experiment


N_K = 3
T_STEP = 0.5
N_STEPS = 20
# True dynamics decay fast; bound caps k at UPPER_K (well below TRUE_K) so the
# optimizer would want to escape if the bound were ignored.
TRUE_K = np.array([5.0, 6.0, 7.0])
LOWER_K = 0.01
UPPER_K = 1.0
SCALING_K = 0.5
SEED_K = np.array([0.5, 0.5, 0.5])
X0 = 10.0


def _build_pe(*, set_bounds_after_setup: bool) -> ParameterEstimator:
    m = do_mpc.model.Model("continuous")
    x = m.set_variable("_x", "x", shape=(N_K, 1))
    k = m.set_variable("_p", "k", shape=(N_K, 1))
    m.set_rhs("x", -k * x)
    for i in range(N_K):
        m.set_meas(f"x_meas_{i}", x[i], meas_noise=True)
    m.setup()

    pe = ParameterEstimator(m, p_est=["k"])
    pe.scaling["_p_est", "k"] = SCALING_K
    if not set_bounds_after_setup:
        pe.bounds["lower", "_p_est", "k"] = LOWER_K
        pe.bounds["upper", "_p_est", "k"] = UPPER_K
    pe.set_default_objective(P_x=0.01 * np.eye(m.n_x), P_v=np.eye(m.n_v))

    t = np.arange(N_STEPS + 1) * T_STEP
    df = pd.DataFrame({"time": t})
    for i in range(N_K):
        df[f"x_meas_{i}"] = X0 * np.exp(-TRUE_K[i] * t)
    exp = Experiment(m)
    exp.set_settings({"collocation_type": "radau", "collocation_deg": 3,
                      "collocation_ni": 1, "t_step": T_STEP})
    exp.setup(data=df, initial_state=np.full((N_K, 1), X0))
    pe.add_experiment(exp)
    pe.setup()

    if set_bounds_after_setup:
        pe.bounds["lower", "_p_est", "k"] = LOWER_K
        pe.bounds["upper", "_p_est", "k"] = UPPER_K

    pe._p_est0["k"] = SEED_K
    return pe


def _assert_bounds_respected(values: np.ndarray, label: str) -> None:
    tol = 1e-4  # IPOPT termination tolerance leaves microscopic slack
    assert np.all(values <= UPPER_K + tol), (
        f"{label}: at least one element of 'k' violated the upper bound "
        f"({UPPER_K}); estimated = {values.tolist()}."
    )
    assert np.all(values >= LOWER_K - tol), (
        f"{label}: at least one element of 'k' violated the lower bound "
        f"({LOWER_K}); estimated = {values.tolist()}."
    )


@pytest.mark.parametrize("set_bounds_after_setup", [False, True])
def test_collocation_respects_scalar_bounds_on_vector_slot(set_bounds_after_setup):
    pe = _build_pe(set_bounds_after_setup=set_bounds_after_setup)
    pe.estimate_parameters(
        method="collocation", auto_save=False, print_identifiability_summary=False
    )
    values = np.array(pe._p_est0["k"]).flatten()
    _assert_bounds_respected(
        values,
        f"collocation (bounds set {'after' if set_bounds_after_setup else 'before'} setup)",
    )


@pytest.mark.parametrize("set_bounds_after_setup", [False, True])
def test_single_shooting_respects_scalar_bounds_on_vector_slot(set_bounds_after_setup):
    pe = _build_pe(set_bounds_after_setup=set_bounds_after_setup)
    pe.estimate_parameters(
        method="single_shooting", auto_save=False, print_identifiability_summary=False
    )
    values = np.array(pe._p_est0["k"]).flatten()
    _assert_bounds_respected(
        values,
        f"single_shooting (bounds set {'after' if set_bounds_after_setup else 'before'} setup)",
    )

if __name__ == '__main__':
    unittest.main()