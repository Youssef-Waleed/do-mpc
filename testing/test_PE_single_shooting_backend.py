"""Tests for the single-shooting parameter-estimation backend.

The backend keeps only the parameters (and optionally each experiment's
initial state) as optimization variables and rolls a differentiable CasADi
integrator forward, fitting a direct weighted least-squares objective with
IPOPT. These tests verify parameter recovery on a pure-ODE model (cvodes) and
a DAE model (idas), agreement with the collocation backend, and the
``estimate_x0`` option.
"""

from __future__ import annotations
import unittest

import numpy as np
import pandas as pd
import do_mpc

from do_mpc.estimator import ParameterEstimator, Experiment


TRUE_K = 0.30
P0 = 10.0
T_STEP = 0.5
N = 40


# -------------------------------------------------------------------------
#  ODE model (cvodes path)
# -------------------------------------------------------------------------
def _ode_model() -> do_mpc.model.Model:
    m = do_mpc.model.Model("continuous")
    P = m.set_variable("_x", "P")
    k = m.set_variable("_p", "k")
    m.set_rhs("P", -k * P)
    m.set_meas("P_meas", P, meas_noise=True)
    m.setup()
    return m


def _ode_data() -> pd.DataFrame:
    t = np.arange(N + 1) * T_STEP
    return pd.DataFrame({"time": t, "P_meas": P0 * np.exp(-TRUE_K * t)})


def _build_ode_pe(estimate_x0: bool = False, x0_seed: float = P0) -> ParameterEstimator:
    m = _ode_model()
    pe = ParameterEstimator(m, p_est=["k"])
    pe.scaling["_p_est", "k"] = TRUE_K
    pe.scaling["_x", "P"] = P0
    pe.bounds["lower", "_p_est", "k"] = 0.001
    pe.bounds["upper", "_p_est", "k"] = 10.0
    pe.bounds["lower", "_x", "P"] = 0.0
    pe.bounds["upper", "_x", "P"] = 100.0
    pe.set_default_objective(P_x=0.01 * np.eye(m.n_x), P_v=np.eye(m.n_v))
    pe.settings.estimate_x0 = estimate_x0

    exp = Experiment(m)
    exp.set_settings({"collocation_type": "radau", "collocation_deg": 3,
                      "collocation_ni": 1, "t_step": T_STEP})
    # Data always starts at the true P0; x0_seed lets a test hand the estimator
    # a deliberately wrong initial state to recover via estimate_x0.
    exp.setup(data=_ode_data(), initial_state=np.array([[x0_seed]]))
    pe.add_experiment(exp)
    pe.setup()
    pe._p_est0["k"] = 0.1
    return pe


# -------------------------------------------------------------------------
#  DAE model (idas path): Q pinned to 2*P, measured channel is Q
# -------------------------------------------------------------------------
def _dae_model() -> do_mpc.model.Model:
    m = do_mpc.model.Model("continuous")
    P = m.set_variable("_x", "P")
    Q = m.set_variable("_z", "Q")
    k = m.set_variable("_p", "k")
    m.set_rhs("P", -k * P)
    m.set_alg("Q_def", Q - 2.0 * P)
    m.set_meas("Q_meas", Q, meas_noise=True)
    m.setup()
    return m


def _build_dae_pe() -> ParameterEstimator:
    m = _dae_model()
    pe = ParameterEstimator(m, p_est=["k"])
    pe.scaling["_p_est", "k"] = TRUE_K
    pe.scaling["_x", "P"] = P0
    pe.bounds["lower", "_p_est", "k"] = 0.001
    pe.bounds["upper", "_p_est", "k"] = 10.0
    pe.set_default_objective(P_x=0.01 * np.eye(m.n_x), P_v=np.eye(m.n_v))

    t = np.arange(N + 1) * T_STEP
    df = pd.DataFrame({"time": t, "Q_meas": 2.0 * P0 * np.exp(-TRUE_K * t)})
    exp = Experiment(m)
    exp.set_settings({"collocation_type": "radau", "collocation_deg": 3,
                      "collocation_ni": 1, "t_step": T_STEP})
    exp.setup(data=df, initial_state=np.array([[P0]]),
              initial_algebraic=np.array([[2.0 * P0]]))
    pe.add_experiment(exp)
    pe.setup()
    pe._p_est0["k"] = 0.1
    return pe


# -------------------------------------------------------------------------
#  Tests
# -------------------------------------------------------------------------
def test_single_shooting_recovers_ode_parameter():
    pe = _build_ode_pe()
    k = pe.estimate_parameters(method="single_shooting", auto_save=False)["k"]
    np.testing.assert_allclose(k, TRUE_K, rtol=1e-3)


def test_single_shooting_matches_collocation_ode():
    k_ss = _build_ode_pe().estimate_parameters(method="single_shooting", auto_save=False)["k"]
    k_co = _build_ode_pe().estimate_parameters(method="collocation", auto_save=False)["k"]
    np.testing.assert_allclose(k_ss, k_co, rtol=1e-3)


def test_single_shooting_recovers_dae_parameter():
    """idas path: measurement reads the algebraic output of the integrator."""
    pe = _build_dae_pe()
    k = pe.estimate_parameters(method="single_shooting", auto_save=False)["k"]
    np.testing.assert_allclose(k, TRUE_K, rtol=1e-3)


def test_single_shooting_estimate_x0_corrects_wrong_initial_state():
    """A deliberately wrong x0 seed (1.5*P0) must be corrected when estimate_x0
    is on, and k still recovers — exercising the t=0 measurement anchor on the
    free initial state. With a fixed (wrong) x0, recovery would be biased."""
    pe = _build_ode_pe(estimate_x0=True, x0_seed=1.5 * P0)
    k = pe.estimate_parameters(method="single_shooting", auto_save=False)["k"]
    np.testing.assert_allclose(k, TRUE_K, rtol=1e-3)


def test_single_shooting_parameter_ordering():
    """Two heterogeneous-magnitude parameters must come back in the right slots.

    Guards the `_p_est0.cat` ordering assumption shared by the scaling vector,
    the initial guess, and the dispatcher's writeback. Asymmetric scales/values
    (k1=2.0, k2=0.05) would surface a transposition that symmetric params hide.
    """
    k1_true, k2_true = 2.0, 0.05
    m = do_mpc.model.Model("continuous")
    P1 = m.set_variable("_x", "P1")
    P2 = m.set_variable("_x", "P2")
    k1 = m.set_variable("_p", "k1")
    k2 = m.set_variable("_p", "k2")
    m.set_rhs("P1", -k1 * P1)
    m.set_rhs("P2", -k2 * P2)
    m.set_meas("P1_meas", P1, meas_noise=True)
    m.set_meas("P2_meas", P2, meas_noise=True)
    m.setup()

    pe = ParameterEstimator(m, p_est=["k1", "k2"])
    pe.scaling["_p_est", "k1"] = k1_true
    pe.scaling["_p_est", "k2"] = k2_true
    pe.scaling["_x", "P1"] = P0
    pe.scaling["_x", "P2"] = P0
    pe.bounds["lower", "_p_est", "k1"] = 0.001
    pe.bounds["upper", "_p_est", "k1"] = 10.0
    pe.bounds["lower", "_p_est", "k2"] = 0.001
    pe.bounds["upper", "_p_est", "k2"] = 10.0
    pe.set_default_objective(P_x=0.01 * np.eye(m.n_x), P_v=np.eye(m.n_v))

    t = np.arange(N + 1) * T_STEP
    df = pd.DataFrame({
        "time": t,
        "P1_meas": P0 * np.exp(-k1_true * t),
        "P2_meas": P0 * np.exp(-k2_true * t),
    })
    exp = Experiment(m)
    exp.set_settings({"collocation_type": "radau", "collocation_deg": 3,
                      "collocation_ni": 1, "t_step": T_STEP})
    exp.setup(data=df, initial_state=np.array([[P0], [P0]]))
    pe.add_experiment(exp)
    pe.setup()
    pe._p_est0["k1"] = 1.0
    pe._p_est0["k2"] = 0.1

    res = pe.estimate_parameters(method="single_shooting", auto_save=False)
    np.testing.assert_allclose(res["k1"], k1_true, rtol=1e-3)
    np.testing.assert_allclose(res["k2"], k2_true, rtol=1e-3)


def test_single_shooting_solver_stats_routed():
    """solver_stats reflects the single-shooting solve, not the collocation backend."""
    pe = _build_ode_pe()
    pe.estimate_parameters(method="single_shooting", auto_save=False)
    assert pe.solver_stats is not None
    assert pe.solver_stats.get("success") is True


def test_single_shooting_estimated_initial_states_none_when_off():
    """With estimate_x0 off, the backend's own clear path yields None (distinct
    from the dispatcher's reset-at-start)."""
    pe = _build_ode_pe(estimate_x0=False)
    pe.estimate_parameters(method="single_shooting", auto_save=False)
    assert pe.estimated_initial_states is None


def test_single_shooting_estimate_x0_surfaces_recovered_state():
    """With estimate_x0 on, the recovered x0 is exposed (physical units here)
    and matches the true P0 the wrong 1.5*P0 seed was corrected toward."""
    pe = _build_ode_pe(estimate_x0=True, x0_seed=1.5 * P0)
    pe.estimate_parameters(method="single_shooting", auto_save=False)
    assert pe.estimated_initial_states is not None
    assert len(pe.estimated_initial_states) == 1
    np.testing.assert_allclose(
        np.asarray(pe.estimated_initial_states[0]).flatten()[0], P0, rtol=1e-2
    )

if __name__ == '__main__':
    unittest.main()