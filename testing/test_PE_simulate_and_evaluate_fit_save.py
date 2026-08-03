"""Tests for ``simulate_and_evaluate_fit(save_results=True)`` autosave.

The post-estimation validation call is the primary use case; callers
can call ``simulate_and_evaluate_fit(val_exps, save_results=True)`` so the
NRMSE-on-validation table persists alongside ``estimation_results.json``.
The pre-fit path (``experiments=None``) may also opt in.

The tests below pin:

- Default behaviour writes nothing.
- ``save_results=True`` writes a JSON file with the expected schema.
- Pre-fit and validation runs produce distinguishable JSON content (the
  experiment list in the JSON differs between the two calls; the file
  name itself overwrites by design).
"""
from __future__ import annotations

import json
from pathlib import Path
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


def _make_sim_result(seed: float) -> SimulationResult:
    """Tiny SimulationResult used to exercise the save path.

    ``seed`` shifts the measurement column so the resulting NRMSE differs
    between pre-fit and validation calls in the distinguishability test.
    """
    return SimulationResult(
        time=np.array([0.0, 1.0, 2.0]),
        y_pred=np.array(
            [
                [1.0, 0.2],
                [1.5, 0.3],
                [2.5, 0.4],
            ]
        ),
        y_meas=np.array(
            [
                [1.0 + seed, 0.2],
                [2.0 + seed, 0.3],
                [3.0 + seed, 0.4],
            ]
        ),
        y_mask=np.ones((3, 2), dtype=int),
        meas_names=["x_meas", "u"],
    )


def _build_estimator() -> ParameterEstimator:
    return ParameterEstimator(
        _build_model_with_input_measurement(), p_est=["k"]
    )


# -------------------------------------------------------------------------
# 1. Default: no JSON written
# -------------------------------------------------------------------------


def test_simulate_and_evaluate_fit_default_no_save(tmp_path, monkeypatch):
    """Default ``save_results=False`` must not touch the filesystem."""
    monkeypatch.chdir(tmp_path)
    estimator = _build_estimator()
    sim_res = _make_sim_result(seed=0.0)
    estimator.simulate_experiments = lambda experiments=None: [sim_res]  # type: ignore[method-assign]

    sim_results, fit_report = estimator.simulate_and_evaluate_fit()

    assert sim_results == [sim_res]
    assert fit_report.per_experiment[0].channel_metrics["x_meas"].nrmse > 0
    # No JSON file in cwd.
    assert not (tmp_path / "validation_results.json").exists()


# -------------------------------------------------------------------------
# 2. save_results=True writes the expected JSON
# -------------------------------------------------------------------------


def test_simulate_and_evaluate_fit_save_results_writes_json(tmp_path, monkeypatch):
    """``save_results=True`` writes ``validation_results.json`` with the
    expected schema."""
    monkeypatch.chdir(tmp_path)
    estimator = _build_estimator()
    sim_res = _make_sim_result(seed=0.0)
    estimator.simulate_experiments = lambda experiments=None: [sim_res]  # type: ignore[method-assign]

    estimator.simulate_and_evaluate_fit(save_results=True)

    json_path = tmp_path / "validation_results.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())

    # Required keys are present
    for key in (
        "timestamp",
        "experiments",
        "settings",
        "estimated_parameters",
        "fixed_parameters",
        "mean_nrmse",
        "per_experiment",
    ):
        assert key in payload, f"missing key: {key}"

    # mean_nrmse matches what the FitReport reports
    assert payload["mean_nrmse"] is not None
    np.testing.assert_allclose(
        payload["mean_nrmse"],
        estimator.evaluate_fit([sim_res]).mean_nrmse,
    )

    # Per-experiment block has channel-level NRMSE entries
    per_exp = payload["per_experiment"]
    assert len(per_exp) == 1
    channels = per_exp[0]["channels"]
    assert "x_meas" in channels
    # The default 'exclude_inputs=True' means 'u' should be filtered out.
    assert "u" not in channels
    assert channels["x_meas"]["nrmse"] is not None
    assert channels["x_meas"]["rmse"] is not None
    assert channels["x_meas"]["n_valid"] == 3

    # Estimated parameters dict is populated (k is in p_est).
    assert "k" in payload["estimated_parameters"]


# -------------------------------------------------------------------------
# 3. Pre-fit and validation runs produce distinguishable content
# -------------------------------------------------------------------------


def test_save_results_pre_fit_and_validation_distinct(tmp_path, monkeypatch):
    """Pre-fit and validation calls produce distinguishable JSON content.

    Both calls write ``validation_results.json`` (overwriting on the
    second call by design — callers own the ordering). The contents
    differ because they use different experiments and therefore
    different NRMSE values. We capture each payload before it is
    overwritten.
    """
    monkeypatch.chdir(tmp_path)
    estimator = _build_estimator()

    pre_fit_sim = _make_sim_result(seed=0.0)
    val_sim = _make_sim_result(seed=0.5)

    # Stub simulate_experiments to switch on whether ``experiments`` is None
    # — that's exactly how the helper distinguishes pre-fit from validation.
    def _fake_simulate(experiments=None):
        if experiments is None:
            return [pre_fit_sim]
        return [val_sim]

    estimator.simulate_experiments = _fake_simulate  # type: ignore[method-assign]

    # Pre-fit run first
    estimator.simulate_and_evaluate_fit(save_results=True)
    pre_fit_payload = json.loads(
        (tmp_path / "validation_results.json").read_text()
    )

    # Then a validation run with a non-None experiments list. The list
    # itself is opaque to the stub, but it triggers the second branch.
    estimator.simulate_and_evaluate_fit(
        experiments=[object()], save_results=True
    )
    validation_payload = json.loads(
        (tmp_path / "validation_results.json").read_text()
    )

    # The two payloads must differ on the NRMSE — pre-fit data and
    # validation data have different residuals by construction.
    pre_nrmse = pre_fit_payload["per_experiment"][0]["channels"]["x_meas"][
        "nrmse"
    ]
    val_nrmse = validation_payload["per_experiment"][0]["channels"]["x_meas"][
        "nrmse"
    ]
    assert pre_nrmse != val_nrmse


def test_save_results_failure_warns_does_not_raise(tmp_path, monkeypatch):
    """If JSON write fails for any reason the helper warns and returns
    the report — the validation run must not crash on save errors."""
    import warnings as _warnings

    monkeypatch.chdir(tmp_path)
    estimator = _build_estimator()
    sim_res = _make_sim_result(seed=0.0)
    estimator.simulate_experiments = lambda experiments=None: [sim_res]  # type: ignore[method-assign]

    # Force failure by monkeypatching the writer to raise.
    def _broken_save(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        ParameterEstimator, "_save_fit_results", _broken_save
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        sim_results, fit_report = estimator.simulate_and_evaluate_fit(
            save_results=True
        )

    assert sim_results == [sim_res]
    # Warning was raised mentioning the validation file
    messages = [str(w.message) for w in caught]
    assert any("validation_results.json" in m for m in messages)

if __name__ == '__main__':
    unittest.main()