from __future__ import annotations
import unittest

import numpy as np
import do_mpc

from do_mpc.estimator._parameterestimator import (
    ParameterEstimator,
    SimulationResult,
)


def _build_model_with_input_measurement() -> do_mpc.model.Model:
    model = do_mpc.model.Model("continuous", "MX")
    x = model.set_variable("_x", "x")
    u = model.set_variable("_u", "u")
    k = model.set_variable("_p", "k")
    model.set_rhs("x", -k * x + u)
    model.set_meas("x_meas", x, meas_noise=True)
    model.set_meas("u", u, meas_noise=True)
    model.setup()
    return model


def test_compute_nrmse_constant_signal_behavior():
    pred = np.array([5.0, 5.0, 5.0])
    meas = np.array([5.0, 5.0, 5.0])
    assert ParameterEstimator.compute_nrmse(pred, meas) == 0.0

    pred_bad = np.array([5.0, 5.0, 5.0])
    meas_bad = np.array([5.1, 5.1, 5.1])
    assert ParameterEstimator.compute_nrmse(pred_bad, meas_bad) == float("inf")


def test_evaluate_fit_excludes_input_measurements_by_default():
    model = _build_model_with_input_measurement()
    estimator = ParameterEstimator(model, p_est=["k"])

    sim_result = SimulationResult(
        time=np.array([0.0, 1.0, 2.0]),
        y_pred=np.array([
            [1.0, 0.2],
            [1.5, 0.3],
            [2.5, 0.4],
        ]),
        y_meas=np.array([
            [1.0, 0.2],
            [2.0, 0.3],
            [3.0, 0.4],
        ]),
        y_mask=np.ones((3, 2), dtype=int),
        meas_names=["x_meas", "u"],
    )

    fit_report = estimator.evaluate_fit([sim_result])

    assert len(fit_report.per_experiment) == 1
    assert set(fit_report.per_experiment[0].channel_metrics.keys()) == {"x_meas"}
    metrics = fit_report.per_experiment[0].channel_metrics["x_meas"]
    np.testing.assert_allclose(metrics.rmse, np.sqrt((0.0**2 + 0.5**2 + 0.5**2) / 3.0))
    np.testing.assert_allclose(metrics.nrmse, metrics.rmse / 2.0)
    assert metrics.n_valid == 3
    assert fit_report.mean_nrmse == metrics.nrmse


def test_simulate_and_evaluate_fit_returns_both_results_and_report():
    model = _build_model_with_input_measurement()
    estimator = ParameterEstimator(model, p_est=["k"])

    sim_result = SimulationResult(
        time=np.array([0.0, 1.0]),
        y_pred=np.array([[1.0, 0.2], [2.0, 0.3]]),
        y_meas=np.array([[1.0, 0.2], [2.5, 0.3]]),
        y_mask=np.ones((2, 2), dtype=int),
        meas_names=["x_meas", "u"],
    )

    estimator.simulate_experiments = lambda experiments=None: [sim_result]  # type: ignore[method-assign]

    sim_results, fit_report = estimator.simulate_and_evaluate_fit()

    assert len(sim_results) == 1
    assert sim_results[0] is sim_result
    assert len(fit_report.per_experiment) == 1
    assert "x_meas" in fit_report.per_experiment[0].channel_metrics

if __name__ == '__main__':
    unittest.main()