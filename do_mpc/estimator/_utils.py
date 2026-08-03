"""Utility functions for the parameter estimator.

Helper functions kept separate from ``Experiment`` and ``ParameterEstimator``
so the core logic stays focused.
"""

import numpy as np
import pandas as pd
import casadi.tools as castools
from typing import Dict, List, Optional, Union


def build_dae(model) -> dict:
    """Build a time-rescaled CasADi DAE dict for use with ``casadi.integrator``.

    The DAE is parameterized over the unit interval: the ODE right-hand side
    is multiplied by a step-length symbol so that integrating from 0 to 1
    advances the physical time by that step. One compiled integrator therefore
    serves every interval length — the step enters as data, not as part of the
    compiled problem. Algebraic equations are instantaneous constraints and
    are not rescaled.

    The integrator's ``p`` input packs ``vertcat(model._p, model._tvp, dt)``
    where ``dt`` is the physical length of the integration interval. The
    integrator's ``u`` input is the model's ``u``. This layout is shared by
    the single-shooting backend and the forward-simulation helper in
    :py:class:`ParameterEstimator`; construct the integrator with time
    horizon ``(0.0, 1.0)``.

    Parameters
    ----------
    model : do_mpc.model.Model
        A configured and setup do-mpc model.

    Returns
    -------
    dict
        DAE dict suitable for ``casadi.integrator(name, plugin, dae, 0.0, 1.0, opts)``.
    """
    w_zero = model._w(0)
    dt = model.sv.sym("dt_intg")
    xdot = model._rhs_fun(model._x, model._u, model._z, model._tvp, model._p, w_zero)
    p_intg = castools.vertcat(
        castools.vertcat(model._p), castools.vertcat(model._tvp), dt
    )
    dae = {
        "x": castools.vertcat(model._x),
        "p": p_intg,
        "u": castools.vertcat(model._u),
        "ode": dt * xdot,
    }
    if model.n_z > 0:
        alg = model._alg_fun(model._x, model._u, model._z, model._tvp, model._p, w_zero)
        dae["z"] = castools.vertcat(model._z)
        dae["alg"] = alg
    return dae


def build_refined_grid(
    t_boundaries: np.ndarray,
    t_extra: np.ndarray,
    rel_tol: float = 1e-9,
) -> tuple:
    """Union of an integration grid with extra time points (e.g. measurement times).

    Integrating over the refined grid places a boundary at every extra time
    point, so states can be read off at those times exactly instead of at the
    nearest original boundary. Each refined interval keeps a pointer to the
    original interval that contains it, for zero-order-hold input/TVP lookup.

    Parameters
    ----------
    t_boundaries : np.ndarray, shape (N+1,)
        Original integration boundaries (sorted, e.g. ``experiment.fe_time``).
    t_extra : np.ndarray, shape (M,)
        Additional time points to insert (e.g. ``experiment.meas_time``).
        Points outside ``[t_boundaries[0], t_boundaries[-1]]`` are ignored.
    rel_tol : float
        Two time points closer than ``rel_tol * max(1, |t_end|)`` are treated
        as identical (the earlier one is kept).

    Returns
    -------
    (t_grid, owner_interval, extra_grid_idx) : tuple
        ``t_grid`` — refined boundaries, shape (K+1,).
        ``owner_interval`` — for each of the K refined intervals, the index of
        the original interval containing it, shape (K,).
        ``extra_grid_idx`` — for each entry of ``t_extra``, the index of its
        boundary in ``t_grid``, shape (M,).
    """
    t_boundaries = np.asarray(t_boundaries, dtype=float)
    t_extra = np.asarray(t_extra, dtype=float)
    t0, tf = t_boundaries[0], t_boundaries[-1]
    tol = rel_tol * max(1.0, abs(tf))

    in_range = t_extra[(t_extra >= t0 - tol) & (t_extra <= tf + tol)]
    t_grid = np.sort(np.concatenate([t_boundaries, np.clip(in_range, t0, tf)]))
    keep = np.ones(len(t_grid), dtype=bool)
    keep[1:] = np.diff(t_grid) > tol
    t_grid = t_grid[keep]

    n_orig = len(t_boundaries) - 1
    owner_interval = np.clip(
        np.searchsorted(t_boundaries, t_grid[:-1] + tol, side="right") - 1,
        0,
        max(n_orig - 1, 0),
    )

    extra_grid_idx = np.array(
        [int(np.argmin(np.abs(t_grid - t_m))) for t_m in t_extra], dtype=int
    )
    return t_grid, owner_interval, extra_grid_idx


def simulate_trajectory(
    model,
    t_intervals: np.ndarray,
    u_seq: np.ndarray,
    tvp_seq: np.ndarray,
    p_full: np.ndarray,
    x0: np.ndarray,
    integrator_cache: dict,
    plugin: str,
    integrator_opts: dict,
    raise_on_failure: bool = False,
) -> Optional[np.ndarray]:
    """Integrate a do-mpc model forward over a sequence of intervals.

    Parameters
    ----------
    model : do_mpc.model.Model
    t_intervals : np.ndarray, shape (N+1,)
        Time boundaries; N intervals are integrated.
    u_seq : np.ndarray, shape (N, n_u)
        Zero-order-hold input for each interval.
    tvp_seq : np.ndarray, shape (N, n_tvp)
        Zero-order-hold TVP for each interval.
    p_full : np.ndarray, shape (n_p,)
        Full model parameter vector (estimated + fixed).
    x0 : np.ndarray, shape (n_x,)
        Initial state.
    integrator_cache : dict
        Persistent integrator cache keyed by plugin name. The integrator is
        time-rescaled (see :func:`build_dae`), so a single compiled instance
        serves every interval length. Mutated in-place; pass the same dict
        across calls to avoid recompiling.
    plugin : str
        CasADi integrator plugin (``'cvodes'`` or ``'idas'``).
    integrator_opts : dict
        Options forwarded to the CasADi integrator (e.g. tolerances).

    Returns
    -------
    np.ndarray, shape (N+1, n_x)
        States at each interval boundary starting from x0, or ``None``
        if integration fails or produces non-finite values.
    """
    n_intervals = len(t_intervals) - 1
    n_tvp = model.n_tvp

    x_traj = np.empty((n_intervals + 1, model.n_x))
    x_traj[0] = x0
    x_current = x0.copy()

    if plugin not in integrator_cache:
        integrator_cache[plugin] = castools.integrator(
            "sim_step", plugin, build_dae(model), 0.0, 1.0, integrator_opts
        )

    for k in range(n_intervals):
        dt = float(t_intervals[k + 1] - t_intervals[k])

        tvp_k = tvp_seq[k] if n_tvp > 0 else np.array([])
        p_intg = np.concatenate([p_full, tvp_k, [dt]])

        try:
            res = integrator_cache[plugin](x0=x_current, p=p_intg, u=u_seq[k].reshape(-1, 1))
            x_next = np.array(res["xf"]).flatten()
        except Exception as exc:
            if raise_on_failure:
                raise RuntimeError(
                    f"integrator raised at interval {k} (t={t_intervals[k]:g} → "
                    f"{t_intervals[k + 1]:g}): {type(exc).__name__}: {exc}"
                ) from exc
            return None

        if not np.all(np.isfinite(x_next)):
            if raise_on_failure:
                raise RuntimeError(
                    f"integrator produced non-finite state at interval {k} "
                    f"(t={t_intervals[k + 1]:g}): x_next={x_next!r}"
                )
            return None

        x_traj[k + 1] = x_next
        x_current = x_next

    return x_traj


def evaluate_measurements(
    model,
    x_traj: np.ndarray,
    t_boundaries: np.ndarray,
    t_meas: np.ndarray,
    u_seq: np.ndarray,
    tvp_seq: np.ndarray,
    p_full: np.ndarray,
) -> np.ndarray:
    """Evaluate the model measurement function at specified time points.

    Looks up the nearest integration boundary for each measurement time
    and evaluates ``model._meas_fun`` (with v=0, noiseless) at the
    corresponding state.

    Parameters
    ----------
    model : do_mpc.model.Model
    x_traj : np.ndarray, shape (N+1, n_x)
        States at t_boundaries, as returned by :func:`simulate_trajectory`.
    t_boundaries : np.ndarray, shape (N+1,)
        Time points corresponding to rows of x_traj.
    t_meas : np.ndarray, shape (M,)
        Times at which to evaluate measurements.
    u_seq : np.ndarray, shape (N, n_u)
        Zero-order-hold inputs (same as used in :func:`simulate_trajectory`).
    tvp_seq : np.ndarray, shape (N, n_tvp)
        Zero-order-hold TVPs.
    p_full : np.ndarray, shape (n_p,)
        Full model parameter vector.

    Returns
    -------
    np.ndarray, shape (M, n_y_all)
        Predicted measurements at each t_meas in the same order as
        ``model.y.keys()[1:]`` (all measurement channels, including
        input pass-throughs).
    """
    n_ivl = len(t_boundaries) - 1
    n_tvp = model.n_tvp
    z_zero = np.zeros(model.n_z) if model.n_z > 0 else np.array([])
    v_zero = np.zeros(model.n_v)

    rows = []
    for t_m in t_meas:
        bi = int(np.argmin(np.abs(t_boundaries - t_m)))
        ivl = min(bi, n_ivl - 1)
        tvp_m = tvp_seq[ivl] if n_tvp > 0 else np.array([])
        y_i = model._meas_fun(x_traj[bi], u_seq[ivl], z_zero, tvp_m, p_full, v_zero)
        rows.append(np.array(y_i).flatten())

    return np.stack(rows) if rows else np.empty((0, len(model.y.keys()) - 1))


def find_closest_time_index(original_time: np.ndarray, t: float) -> int:
    """Find the index of the latest time point not greater than t.

    Uses binary search (np.searchsorted) for efficient lookup.

    Parameters
    ----------
    original_time : np.ndarray
        Sorted array of original time points
    t : float
        Target time point

    Returns
    -------
    int
        Index of the closest time point
    """
    idx = np.searchsorted(original_time, t, side="right") - 1
    if idx < 0:
        idx = 0
    return idx


def validate_data_columns(
    data: pd.DataFrame,
    y_names: List[str],
    u_names: List[str],
    tvp_names: List[str],
    experiment_type: str,
) -> dict:
    """Validate that all required columns are present in the data.

    Parameters
    ----------
    data : pd.DataFrame
        The experiment data
    y_names : List[str]
        Measurement column names
    u_names : List[str]
        Input column names
    tvp_names : List[str]
        Time-varying parameter column names
    experiment_type : str
        Type of experiment ('dynamic', etc.)

    Returns
    -------
    dict
        Structured missing-column report with keys
        ``missing_outputs``, ``missing_inputs``, ``missing_tvps``,
        ``unmatched_columns``.

    Raises
    ------
    AssertionError
        If required input/TVP/time columns are missing, or if the DataFrame
        carries columns that don't correspond to any model measurement,
        input, or TVP.
    """
    missing_outputs = [col for col in y_names if col not in data.columns]
    missing_inputs = [col for col in u_names if col not in data.columns]
    missing_tvps = [col for col in tvp_names if col not in data.columns]
    if missing_inputs:
        raise AssertionError(f"Input columns missing in the data: {missing_inputs}")
    if missing_tvps:
        raise AssertionError(f"TVP columns missing in the data: {missing_tvps}")

    if experiment_type == "dynamic" and "time" not in data.columns:
        raise AssertionError("Time column is missing in the data.")

    known = set(y_names) | set(u_names) | set(tvp_names) | {"time"}
    unmatched_columns = [col for col in data.columns if col not in known]
    if unmatched_columns:
        raise AssertionError(
            f"DataFrame has columns {unmatched_columns} that don't correspond "
            f"to any model measurement, input, or TVP — either remove them "
            f"from the DataFrame or register a matching `set_meas` in the model."
        )

    return {
        "missing_outputs": missing_outputs,
        "missing_inputs": missing_inputs,
        "missing_tvps": missing_tvps,
        "unmatched_columns": unmatched_columns,
    }


def consolidate_nearby_timepoints(
    timepoints: np.ndarray, min_distance: float = 1e-5
) -> np.ndarray:
    """Consolidate timepoints that are very close to each other.

    Groups nearby timepoints into clusters and takes the minimum
    of each cluster to avoid ambiguous assignment to discretization elements.

    Parameters
    ----------
    timepoints : np.ndarray
        Array of timepoints to consolidate
    min_distance : float, optional
        Minimum distance between timepoints, by default 1e-5

    Returns
    -------
    np.ndarray
        Consolidated timepoints
    """
    if len(timepoints) <= 1:
        return timepoints

    sorted_times = np.sort(timepoints)
    diffs = np.diff(sorted_times)

    # Find clusters of nearby points
    cluster_starts = np.where(diffs > min_distance)[0] + 1
    cluster_starts = np.insert(cluster_starts, 0, 0)

    if len(cluster_starts) > 1:
        cluster_ends = np.append(cluster_starts[1:] - 1, len(sorted_times) - 1)
    else:
        cluster_ends = np.array([len(sorted_times) - 1])

    consolidated_times = []
    for start, end in zip(cluster_starts, cluster_ends):
        if start == end:
            consolidated_times.append(sorted_times[start])
        else:
            # Use the min time point of the cluster (see comment in original code
            # about why mean is problematic for discretization assignment)
            consolidated_times.append(np.min(sorted_times[start : end + 1]))

    return np.array(consolidated_times)


def interpolate_signal_at_index(
    data: pd.DataFrame, signal_name: str, idx: int
) -> float:
    """Get a signal value at a given index, with NaN backward/forward fill.

    Parameters
    ----------
    data : pd.DataFrame
        The experiment data
    signal_name : str
        Column name of the signal
    idx : int
        Index into the DataFrame

    Returns
    -------
    float
        The signal value (0.0 if all values are NaN)
    """
    value = data[signal_name].iloc[idx]
    if not np.isnan(value):
        return value

    # Try backward fill
    j = idx
    while j >= 0 and np.isnan(data[signal_name].iloc[j]):
        j -= 1
    if j >= 0:
        return data[signal_name].iloc[j]

    # Try forward fill
    j = idx + 1
    while j < len(data) and np.isnan(data[signal_name].iloc[j]):
        j += 1
    if j < len(data):
        return data[signal_name].iloc[j]

    return 0.0
