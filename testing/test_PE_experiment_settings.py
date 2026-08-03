"""Tests for Experiment.set_settings defaults and validation."""

from __future__ import annotations
import unittest

import do_mpc
import pandas as pd
import pytest

from do_mpc.estimator._experiment import Experiment


def _make_model():
    model = do_mpc.model.Model("continuous")
    x = model.set_variable("_x", "x")
    model.set_rhs("x", -x)
    model.set_meas("x", x)
    model.setup()
    return model


def test_set_settings_defaults_collocation_ni_to_one():
    exp = Experiment(_make_model())
    exp.set_settings(
        {
            "collocation_type": "radau",
            "collocation_deg": 3,
            "t_step": 1.0,
        }
    )
    assert exp.settings.collocation_ni == 1
    assert exp._settings["collocation_ni"] == 1


def test_set_settings_rejects_unknown_keys():
    exp = Experiment(_make_model())
    with pytest.raises(AssertionError, match="Unknown experiment setting"):
        exp.set_settings(
            {
                "collocation_type": "radau",
                "collocation_deg": 3,
                "t_step": 1.0,
                "typo_key": 1,
            }
        )


def test_set_settings_requires_dynamic_core_keys():
    exp = Experiment(_make_model())
    with pytest.raises(AssertionError, match="Time step is required"):
        exp.set_settings(
            {
                "collocation_type": "radau",
                "collocation_deg": 3,
            }
        )


def test_setup_excludes_missing_measurement_columns_without_nan_injection():
    """Missing y-measurement channels are excluded from the NLP, not dummy-patched."""
    model = do_mpc.model.Model("continuous")
    x = model.set_variable("_x", "x")
    u = model.set_variable("_u", "u")
    model.set_rhs("x", -x + u)
    model.set_meas("x", x)
    model.set_meas("u", u)
    model.setup()

    exp = Experiment(model)
    exp.set_settings(
        {
            "collocation_type": "radau",
            "collocation_deg": 3,
            "t_step": 1.0,
        }
    )

    df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "u": [0.0, 0.0, 0.0],
        }
    )

    with pytest.warns(UserWarning, match="has been excluded from the estimation objective"):
        exp.setup(data=df, initial_state=[[1.0]])

    assert exp.ignored_y_names == ["x"]
    # No dummy NaN column injected into self.data
    assert "x" not in exp.data.columns
    # Mask for the excluded channel must be all-zero
    y_idx = list(exp.y_names).index("x")
    assert exp.y_mask[:, y_idx].sum() == 0


def test_setup_rejects_dataframe_with_unmatched_columns():
    """DataFrame columns that don't pair with any model meas/input/tvp must error."""
    model = do_mpc.model.Model("continuous")
    x = model.set_variable("_x", "x")
    u = model.set_variable("_u", "u")
    model.set_rhs("x", -x + u)
    model.set_meas("x", x)
    model.set_meas("u", u)
    model.setup()

    exp = Experiment(model)
    exp.set_settings(
        {
            "collocation_type": "radau",
            "collocation_deg": 3,
            "t_step": 1.0,
        }
    )

    df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "u": [0.0, 0.0, 0.0],
            "x": [1.0, 0.5, 0.25],
            "speed_actual": [10.0, 11.0, 12.0],  # orphan: no matching set_meas
        }
    )

    with pytest.raises(AssertionError, match="speed_actual"):
        exp.setup(data=df, initial_state=[[1.0]])

if __name__ == '__main__':
    unittest.main()
