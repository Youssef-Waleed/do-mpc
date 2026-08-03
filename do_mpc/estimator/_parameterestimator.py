import json
import numpy as np
import casadi as ca
import casadi.tools as castools
from do_mpc.tools._casstructure import _SymVar
import do_mpc
from pathlib import Path
from typing import Any, Union, Tuple, List, Dict, Callable, Optional
import copy
from dataclasses import dataclass, field
import warnings
from do_mpc.estimator._plotting import (
    plot_results as _plot_results_fn,
)
from do_mpc.estimator._utils import (
    build_refined_grid,
    evaluate_measurements,
    simulate_trajectory,
)
from do_mpc.estimator._struct_utils import dict_to_struct_column

from do_mpc.batch_optimizer import BatchOptimizer

class _TrackedPTemplate(castools.structure3.DMStruct):
    """DM-struct returned by :py:meth:`ParameterEstimator.get_p_template` that
    records which keys the caller has explicitly assigned via
    ``template[key] = value``. :py:meth:`ParameterEstimator.set_p_fun` uses
    the record to detect forgotten parameter assignments — without tracking,
    an unset entry silently stays at the default zero, and IPOPT crashes at
    iteration 0 with ``Invalid_Number_Detected`` without naming the missing
    key.

    Bulk assignment via ``template.master = ...`` does not register
    individual keys; callers using that path must announce the assigned
    keys explicitly via :py:meth:`mark_keys_set`.
    """

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._set_keys.add(key)
        elif isinstance(key, tuple) and key and isinstance(key[0], str):
            self._set_keys.add(key[0])
        super().__setitem__(key, value)

    def mark_keys_set(self, keys) -> None:
        self._set_keys.update(keys)

def _make_tracked_p_template(p_set_struct) -> _TrackedPTemplate:
    template = p_set_struct(0)
    template.__class__ = _TrackedPTemplate
    object.__setattr__(template, "_set_keys", set())
    return template

@dataclass
class ParameterEstimatorSettings:
    """Settings for :py:class:`ParameterEstimator`.

    The :py:class:`ParameterEstimator` automatically creates an instance of type :py:class:`ParameterEstimatorSettings` and adds it to its class attributes.

    Example to change settings:

        ::

            estimator.settings.n_horizon = 20

    Note:
        Settings cannot be updated after calling :py:meth:`ParameterEstimator.setup`.
    """

    meas_from_data: bool = False
    state_discretization: str = "collocation"
    """Default option to retrieve past measurements for the optimization problem."""
    nl_cons_check_colloc_points: bool = False
    """For orthogonal collocation choose whether the bounds set with :py:meth:`set_nl_cons` are evaluated once per finite Element or for each collocation point."""
    nl_cons_single_slack: bool = False
    """If ``True``, soft-constraints set with :py:func:`set_nl_cons` introduce only a single slack variable for the entire horizon."""
    cons_check_colloc_points: bool = True
    """For orthogonal collocation choose whether the linear bounds set with :py:attr:`bounds` are evaluated once per finite Element or for each collocation point."""
    store_full_solution: bool = False
    """Choose whether to store the full solution of the optimization problem."""
    store_lagr_multiplier: bool = True
    """Choose whether to store the lagrange multipliers of the optimization problem."""
    store_solver_stats: List[str] = field(default_factory=lambda: ["success", "t_wall_total"])
    """Choose which solver statistics to store. Must be a list of valid statistics."""
    nlpsol_opts: Dict = field(
        default_factory=lambda: {
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.print_level": 3,
            "ipopt.print_frequency_iter": 50,
        }
    )
    """Dictionary with options for the CasADi solver call ``nlpsol`` with plugin ``ipopt``.

    Defaults are intentionally concise for terminal workflows:
    suppress the IPOPT banner / timing boilerplate, keep a low print level,
    and avoid per-iteration spam on short solves.
    """

    # --- Single-shooting backend ---
    integrator_plugin: str = "auto"
    """CasADi integrator plugin for the single-shooting backend. ``'auto'``
    selects ``'idas'`` for DAE models (``n_z > 0``) and ``'cvodes'`` for pure
    ODE models. Can be overridden to ``'cvodes'``, ``'idas'``, or
    ``'collocation'``."""
    integrator_opts: Dict = field(
        default_factory=lambda: {"abstol": 1e-8, "reltol": 1e-8}
    )
    """Options forwarded to the CasADi integrator in the single-shooting
    backend (e.g. tolerances). Defaults of ``1e-8`` keep the integration
    error well below IPOPT's convergence tolerances; looser values inject
    noise into the objective gradient and slow or stall the optimizer.
    Zero-order-hold input steps are safe at any tolerance because each
    integrator call spans a single constant-input interval."""
    estimate_x0: bool = False
    """Single-shooting only. If ``True``, each experiment's initial state is
    added as a free optimization variable (seeded at ``experiment.initial_state``,
    bounded by the state bounds) instead of being pinned. No arrival-cost prior
    is applied."""

    def check_for_mandatory_settings(self) -> None:
        return None

@dataclass
class SimulationResult:
    """Result of forward-simulating an experiment with estimated parameters.

    Holds predicted and measured outputs at the experiment's measurement times,
    along with the validity mask.  Used for evaluating model fit on validation
    or test data after parameter estimation.
    """

    time: np.ndarray
    """Measurement time points, shape ``(n_meas,)``."""
    y_pred: np.ndarray
    """Predicted measurements from forward simulation, shape ``(n_meas, n_y)``."""
    y_meas: np.ndarray
    """Actual measurements from the experiment data, shape ``(n_meas, n_y)``."""
    y_mask: np.ndarray
    """Binary validity mask (1 = valid, 0 = missing/NaN), shape ``(n_meas, n_y)``."""
    meas_names: list
    """Measurement channel names (``experiment.all_meas_names``)."""


@dataclass
class ChannelFitMetrics:
    """Per-channel fit metrics for a simulated experiment."""

    rmse: float
    """Root mean square error on valid samples only."""
    nrmse: float
    """Range-normalized RMSE on valid samples only."""
    n_valid: int
    """Number of valid sample pairs used for the metric."""


@dataclass
class ExperimentFitReport:
    """Fit metrics for one simulated experiment."""

    experiment_index: int
    """0-based experiment index within the evaluated list."""
    channel_metrics: Dict[str, ChannelFitMetrics]
    """Per-channel fit metrics keyed by measurement name."""


@dataclass
class FitReport:
    """Structured fit report derived from ``SimulationResult`` objects."""

    per_experiment: List[ExperimentFitReport]
    """Per-experiment fit metrics."""

    @property
    def finite_nrmse(self) -> List[float]:
        """Flattened list of finite channel NRMSE values."""
        values: List[float] = []
        for exp_report in self.per_experiment:
            for metrics in exp_report.channel_metrics.values():
                if np.isfinite(metrics.nrmse):
                    values.append(float(metrics.nrmse))
        return values

    @property
    def mean_nrmse(self) -> Optional[float]:
        """Mean finite channel NRMSE, or ``None`` if none are available."""
        values = self.finite_nrmse
        if not values:
            return None
        return float(np.mean(values))

    def print_summary(self, label: Optional[str] = None) -> None:
        """Print mean and per-channel NRMSE for every experiment.

        Use this in place of writing your own loop over ``per_experiment`` —
        the format is uniform pre- and post-estimation so attempts can be
        compared at a glance.
        """
        header = "Fit summary" if label is None else f"Fit summary — {label}"
        mean = self.mean_nrmse
        mean_str = f"{mean:.3f}" if mean is not None else "n/a"
        print(f"{header}: mean NRMSE = {mean_str}")
        for exp_report in self.per_experiment:
            print(f"  experiment {exp_report.experiment_index}:")
            for ch_name, m in exp_report.channel_metrics.items():
                nrmse_str = f"{m.nrmse:.3f}" if np.isfinite(m.nrmse) else "n/a"
                print(f"    {ch_name:40s} NRMSE = {nrmse_str}")


class ParameterEstimator(BatchOptimizer):
    """Parameter estimator using orthogonal collocation + IPOPT.
    
        The ODE is discretized via orthogonal collocation and the resulting NLP
        is solved with IPOPT. Multi-experiment estimation is supported.
    
        **Configuration and setup:**
    
        1. Create the estimator with model and parameter names to estimate.
        2. Add experiments with :py:func:`add_experiment`.
        3. Set the objective with :py:func:`set_objective` or :py:func:`set_default_objective`.
        4. Set bounds and scaling as needed.
        5. Optionally set non-linear constraints with :py:func:`set_nl_cons`.
        6. Set the parameter function with :py:func:`get_p_template` and :py:func:`set_p_fun`.
        7. Call :py:func:`setup` to finalize.
        8. Call :py:func:`estimate_parameters` to run estimation.
    
        Args:
            model: A configured and setup :py:class:`do_mpc.model.Model`
            p_est: List with names of parameters (``_p``) defined in ``model``
    """

    def __init__(
        self,
        model: Union[do_mpc.model.Model, do_mpc.model.LinearModel],
        p_est: Union[str, List[str]] = [],
    ):
        super().__init__(model)
        
        # Initialize structures for bounds, scaling, initial values
        self._x_lb = self.model._x(-np.inf)
        self._x_ub = self.model._x(np.inf)
        
        self._u_lb = self.model._u(-np.inf)
        self._u_ub = self.model._u(np.inf)
        
        self._z_lb = self.model._z(-np.inf)
        self._z_ub = self.model._z(np.inf)
        
        self._x_scaling = self.model._x(1.0)
        self._x_offset = self.model._x(0.0)
        self._u_scaling = self.model._u(1.0)
        self._u_offset = self.model._u(0.0)
        self._z_scaling = self.model._z(1.0)
        self._z_offset = self.model._z(0.0)
        self._p_scaling = self.model._p(1.0)
        self._p_offset = self.model._p(0.0)
        
        self.experiment_list = []
        
        # Dummy variables for bounds of all optimization variables
        self._lb_opt_x = None
        self._ub_opt_x = None
        
        # Lists for further non-linear constraints (optional)
        self.nl_cons_list = [
            {"expr_name": "default", "expr": castools.DM([]), "ub": castools.DM([])}
        ]
        self.slack_vars_list = [
            {"slack_name": "default", "shape": 0, "ub": castools.DM([]), "penalty": 0}
        ]
        self.slack_cost = 0
        
        # Initialize structure to hold the optimal solution and initial guess:
        self._opt_x_num = None
        # Initialize structure to hold the parameters for the optimization problem:
        self._opt_p_num = None
        
        # Initialize settings class
        self.settings = ParameterEstimatorSettings()
        
        if isinstance(p_est, str):
            p_est_list = [p_est]
        elif isinstance(p_est, list):
            p_est_list = p_est
        if len(p_est_list) == 0:
            raise ValueError(
                f"No parameters to estimate. Please provide a list of parameter names. \n The following are available: {model._p.keys()[1:]}"
            )
        
        # Create separate structs for estimated and fixed parameters
        _p = model._p
        self._p_est = self.model.sv.sym_struct(
            [castools.entry("default", shape=(0, 1))]
            + [castools.entry(p_i, shape=_p[p_i].shape) for p_i in _p.keys() if p_i in p_est_list]
        )
        self._p_set = self.model.sv.sym_struct(
            [castools.entry(p_i, shape=_p[p_i].shape) for p_i in _p.keys() if p_i not in p_est_list]
        )
        
        # Enable to "unite" _p_est and _p_set to _p
        p_cat = castools.vertcat(_p)
        _p_subs = []
        for name in _p.keys():
            if name in self._p_est.keys():
                _p_subs.append(self._p_est[name])
            elif name in self._p_set.keys():
                _p_subs.append(self._p_set[name])
        
        p_cat = castools.substitute(p_cat, _p, castools.vertcat(*_p_subs))
        
        # Function to obtain full set of parameters from the separate structs
        self._p_cat_fun = castools.Function("p_cat_fun", [self._p_est, self._p_set], [p_cat])
        
        self.n_p_est = self._p_est.shape[0]
        self.n_p_set = self._p_set.shape[0]
        
        # Initialize additional structures
        self._p_est_scaling = self._p_est(1.0)
        self._p_est_offset = self._p_est(0.0)
        self._p_set_scaling = self._p_set(1.0)
        self._p_set_offset = self._p_set(0.0)
        
        self._p_est_lb = self._p_est(-np.inf)
        self._p_est_ub = self._p_est(np.inf)
        
        self._p_est0 = self._p_est(0.0)
        
        # Per-parameter optimizer-space transform ('linear' by default).
        # Log-flagged slots are optimized as z = log(p / scaling), i.e.
        # p = scaling * exp(z); the flag covers every element of a vector
        # slot. _p_est_log_mask is the per-element view aligned with
        # _p_est.cat; _p_log_mask_full (over the concatenated _p vector) is
        # rebuilt in _refresh_p_transform_metadata.
        self._p_est_transform: Dict[str, str] = {}
        self._p_est_log_mask = np.zeros(self.n_p_est, dtype=bool)
        self._p_log_mask_full = np.zeros(self.model._p.shape[0], dtype=bool)
        self._p_est_element_names: List[str] = []
        for key in self._p_est.keys():
            if key == "default":
                continue
            size = len(self._p_est.f[key])
            if size == 1:
                self._p_est_element_names.append(key)
            else:
                self._p_est_element_names.extend(
                    f"{key}[{i}]" for i in range(size)
                )
        
        # Introduce aliases for symbolic variables
        self._y_meas = self.model._y
        self._y_weight = self.model._y
        
        self._x_prev = copy.copy(self.model._x)
        self._x = self.model._x
        
        self._u = self.model._u
        self._z = self.model._z
        
        self._p = self.model._p
        
        self._w = self.model._w
        self._v = self.model._v
        
        self._tvp = self.model._tvp
        
        # Flags are checked when calling .setup.
        self.flags.update({
            "set_p_fun": False,
            "set_objective": False,
        })
        
        self._P_v_default = None
        self._last_identifiability_report = None
        self._last_solver_outcome: Optional[Dict[str, Any]] = None
        
        # Add flags to track which NLP is compiled to satisfy test constraints[cite: 13]
        self.flags.update({
            "collocation_setup": False,
            "single_shooting_setup": False,
            "set_p_fun": False,
            "set_objective": False,
        })
        self._last_identifiability_report = None
        self._last_solver_outcome = None
        self.estimated_initial_states = None


    # ---------------------------------------------------------------
    #  p_est0 — initial guesses for estimated parameters
    # ---------------------------------------------------------------

    @property
    def p_est0(self) -> Dict[str, float]:
        """Current initial guesses for estimated parameters as ``{name: value}``.

        Returns:
            Dict mapping each estimated parameter name to its current guess.
        """
        return self._estimated_params_dict()

    @p_est0.setter
    def p_est0(self, value: Dict[str, float]) -> None:
        """Set initial guesses from a ``{name: value}`` dict.

        Args:
            value: Dict mapping parameter names to initial-guess values. The
                dict must cover every estimated parameter exactly — missing
                or unknown keys raise :class:`ValueError` (same format as
                :func:`dict_to_struct_column`).
        """
        col = dict_to_struct_column(value, self._p_est, label="p_est0")
        self._p_est0.master = castools.DM(col)

    def _validate_transform_config(self) -> None:
        """Check the configuration of every log-scale parameter.

        A log-scale parameter needs a strictly positive, finite bounds
        bracket (the opt-in targets parameters whose plausible range spans
        decades — that range is stated through ``bounds``), strictly
        positive scaling, and zero offset. Runs at :meth:`setup` and again
        at every :meth:`estimate_parameters` since bounds can change in
        between. Initial-guess positivity is enforced separately where the
        guess is mapped to optimizer coordinates (:meth:`set_initial_guess`).
        """
        errors: List[str] = []
        for name, mode in self._p_est_transform.items():
            if mode != "log":
                continue
            lb = np.asarray(self._p_est_lb[name]).reshape(-1)
            ub = np.asarray(self._p_est_ub[name]).reshape(-1)
            sc = np.asarray(self._p_est_scaling[name]).reshape(-1)
            off = np.asarray(self._p_est_offset[name]).reshape(-1)
            for i in range(lb.size):
                el = name if lb.size == 1 else f"{name}[{i}]"
                if not (np.isfinite(lb[i]) and lb[i] > 0):
                    errors.append(
                        f"  {el}: lower bound {lb[i]!r} — log-scale "
                        f"parameters require a strictly positive, finite "
                        f"lower bound (bounds['lower', '_p_est', {name!r}])."
                    )
                if not (np.isfinite(ub[i]) and ub[i] > 0):
                    errors.append(
                        f"  {el}: upper bound {ub[i]!r} — log-scale "
                        f"parameters require a strictly positive, finite "
                        f"upper bound (bounds['upper', '_p_est', {name!r}])."
                    )
                if np.isfinite(lb[i]) and np.isfinite(ub[i]) and lb[i] >= ub[i]:
                    errors.append(
                        f"  {el}: lower bound {lb[i]!r} >= upper bound {ub[i]!r}."
                    )
                if not sc[i] > 0:
                    errors.append(
                        f"  {el}: scaling {sc[i]!r} — log-scale parameters "
                        f"require strictly positive scaling (p = scaling * exp(z))."
                    )
                if off[i] != 0.0:
                    errors.append(
                        f"  {el}: offset {off[i]!r} — log-scale parameters "
                        f"cannot carry an affine offset."
                    )
        if errors:
            raise ValueError(
                "Invalid log-scale parameter configuration:\n" + "\n".join(errors)
            )

    # ---------------------------------------------------------------
    #  Experiment management
    # ---------------------------------------------------------------

    def _check_duplicate_meas_expressions(self) -> None:
        """Detect set_meas calls that pin the same CasADi expression twice.

        Two measurements registered against structurally-equal expressions
        each contribute a constraint to the estimation NLP. When both carry
        ``meas_noise=False`` they would normally add two redundant *hard*
        equality constraints and over-specify the problem — except for the
        ``_u``-pass-through special case, where the collocation backend
        drops the equality rows and pins ``_u[k]`` via bounds (idempotent
        under repetition). Cases:

        * Both rows are ``_u``-pass-through on the same ``_u`` symbol with
          ``meas_noise=False`` → harmless under the pass-through fix, emit
          a UserWarning so callers learn about the redundancy without
          blocking estimation or evaluator rebuilds.
        * Both ``meas_noise=False`` on a non-``_u`` expression → still
          over-specifying, raise.
        * Mixed ``meas_noise`` → the noise-free side hard-pins; warn.

        Only fires for models that maintain a ``_meas_registry`` attribute;
        plain do-mpc models lack it and pass through silently.
        """
        meas_registry = getattr(self.model, "_meas_registry", None)
        if not meas_registry:
            return
        u_names = {n for n in self.model.u.keys() if n != "default"}

        def _passthrough_u_name(expr) -> Optional[str]:
            """Return the ``_u`` symbol name if ``expr`` is exactly a single
            ``_u`` symbol, else ``None``."""
            try:
                if not expr.is_symbolic():
                    return None
                name = expr.name()
            except Exception:
                return None
            return name if name in u_names else None

        items = list(meas_registry.items())
        errors: List[str] = []
        warning_msgs: List[str] = []
        for i, (name_a, entry_a) in enumerate(items):
            expr_a, _unit_a, mn_a = entry_a[0], entry_a[1], entry_a[2]
            for name_b, entry_b in items[i + 1:]:
                expr_b, _unit_b, mn_b = entry_b[0], entry_b[1], entry_b[2]
                try:
                    # Deep structural comparison; the depth arg must be int.
                    # 2**30 dwarfs any plausible CasADi AST depth (a few
                    # hundred at most); ``ca.inf`` is a Python float and
                    # would raise a type error here.
                    same_expr = ca.is_equal(expr_a, expr_b, 2**30)
                except Exception:
                    same_expr = False
                if not same_expr:
                    continue
                label = (
                    f"measurements '{name_a}' and '{name_b}' are registered "
                    f"with structurally equal expressions"
                )
                # Special case: both are _u-passthrough on the same _u symbol
                # with meas_noise=False. _detect_u_passthrough_meas + the
                # collocation backend together drop the per-step equality
                # rows and pin _u[k] via bounds; pinning the same value twice
                # is a no-op, so the duplication is operationally harmless.
                u_name_a = _passthrough_u_name(expr_a)
                u_name_b = _passthrough_u_name(expr_b)
                both_passthrough = (
                    u_name_a is not None
                    and u_name_a == u_name_b
                    and mn_a is False
                    and mn_b is False
                )
                if both_passthrough:
                    warning_msgs.append(
                        f"{label} — both are pure `_u` pass-throughs on "
                        f"'{u_name_a}' with meas_noise=False. The framework "
                        f"collapses these to a single bounds-pin on _u[k]; "
                        f"estimation is unaffected, but the y vector carries "
                        f"a redundant row."
                    )
                    continue
                if mn_a is False and mn_b is False:
                    errors.append(
                        f"{label}, both with meas_noise=False — this adds two "
                        f"redundant hard equality constraints to the estimation "
                        f"NLP and over-specifies the problem. Drop one of the "
                        f"two registrations or set meas_noise=True on one."
                    )
                else:
                    warning_msgs.append(
                        f"{label} (meas_noise={mn_a!r} for '{name_a}', "
                        f"meas_noise={mn_b!r} for '{name_b}'). The duplication "
                        f"is wasteful — consider dropping the redundant entry."
                    )
        for msg in warning_msgs:
            warnings.warn(msg, UserWarning, stacklevel=2)
        if errors:
            raise ValueError(
                "Over-specified estimation problem from duplicate measurements:\n"
                "  - " + "\n  - ".join(errors)
            )

    def _detect_u_passthrough_meas(self) -> List[Tuple[int, str]]:
        """Identify set_meas channels that are pure `_u` pass-throughs.

        A channel qualifies when its measurement expression evaluates to a
        single `_u` variable (no other symbolic dependencies, unit coefficient,
        zero offset) AND the channel was registered with ``meas_noise=False``.
        The estimator uses this list to skip per-step equality constraints
        and instead pin `_u[k]` directly to FE-aligned data via bounds (see
        CollocationBackend._update_bounds).

        The qualification is checked via Jacobian sparsity + numeric
        evaluation rather than CasADi symbol-name comparison, because the
        substituted MX expression returned by ``_meas_fun`` is a vertsplit
        node whose ``.name()`` and ``is_equal`` checks no longer match the
        original ``model._u[name]`` MX node. Jacobian-based testing works
        identically for SX and MX models.

        Returns a list of ``(y_index, u_name)`` pairs where ``y_index`` is the
        channel's index in ``model.y.keys()[1:]`` (i.e. the row of the
        measurement vector seen by the NLP).
        """
        passthrough: List[Tuple[int, str]] = []
        u_names = [n for n in self.model.u.keys() if n != "default"]
        if not u_names:
            return passthrough

        m = self.model
        y_sym = m._meas_fun(m._x, m._u, m._z, m._tvp, m._p, m._v)
        y_names = [n for n in m.y.keys() if n != "default"]
        other_vars = [m._x, m._z, m._tvp, m._p, m._v]
        zero_args = (
            np.zeros(m.n_x), np.zeros(m.n_u), np.zeros(m.n_z),
            np.zeros(m.n_tvp), np.zeros(m.n_p), np.zeros(m.n_v),
        )

        for y_idx, y_name in enumerate(y_names):
            row = y_sym[y_idx]
            if any(var.shape[0] > 0 and ca.jacobian(row, var).nnz() > 0 for var in other_vars):
                continue
            J_u = ca.jacobian(row, m._u)
            if J_u.nnz() != 1:
                continue
            col = int(J_u.sparsity().get_col()[0])
            u_name = u_names[col]
            f_J = ca.Function("J", [m._x, m._u, m._z, m._tvp, m._p, m._v], [J_u])
            J_val = np.array(f_J(*zero_args)).flatten()
            if not np.isclose(J_val[col], 1.0):
                continue
            f_const = ca.Function("c", [m._x, m._u, m._z, m._tvp, m._p, m._v], [row])
            const_val = float(np.array(f_const(*zero_args)).flatten()[0])
            if not np.isclose(const_val, 0.0):
                continue
            registry = getattr(self.model, "_meas_registry", None)
            if registry is not None:
                entry = registry.get(y_name)
                if entry is not None and len(entry) >= 3 and entry[2] is True:
                    continue
            passthrough.append((y_idx, u_name))
        return passthrough

    def setup(self):
            """Finalize the estimator configuration and build the estimation problem.
    
            This creates the backend (collocation by default) and prepares
            the optimization/estimation problem.
            """
            # --- Reject invalid log-transform configuration early ---
            self._validate_transform_config()
    
            # --- Reject duplicate-expression measurements early ---
            self._check_duplicate_meas_expressions()
    
            # --- Identify _u pass-through measurements ---
            # When the user calls model.set_meas(name, model.u[name], meas_noise=False)
            # to register an external _u (a hard requirement of Experiment), the
            # measurement is *not* informational — its sole purpose is to confirm the
            # input is logged. This is a "replay verbatim from CSV" channel and is
            # explicitly excluded from σ_y.
            #
            # Without special handling, the collocation backend emits one equality
            # constraint per measurement timestep, pinning the free `_u[k]` variable
            # to data. With multiple measurements per FE (n_meas > n_horizon, common
            # whenever the data has any sample-to-sample jitter that crosses
            # change_threshold) this over-constrains the NLP — IPOPT returns
            # Infeasible_Problem_Detected on small cases and TOO_FEW_DOF at scale.
            #
            # The fix: detect these channels and (a) drop their per-measurement
            # equality rows; (b) pin `_u[k]` directly to FE-aligned data via tight
            # bounds. The equality count drops from n_meas to n_horizon per channel
            # and "replay verbatim" actually happens.
            self._u_passthrough_meas = self._detect_u_passthrough_meas()
            self._u_passthrough_y_indices = [
                y_idx for (y_idx, _u_name) in self._u_passthrough_meas
            ]
            self._u_passthrough_keep_y_indices = [
                i for i in range(self.model.n_y)
                if i not in self._u_passthrough_y_indices
            ]
    
            # --- Structural identifiability check (optional, removable) ---
            try:
                from do_mpc.estimator._identifiability import (
                    check_structural_identifiability,
                    warn_unidentifiable,
                )
    
                p_est_names = [k for k in self._p_est.keys() if k != "default"]
                if self.experiment_list:
                    # Drop `_u`-pass-through channels from the "true measurement"
                    # set: the data they carry is the input we already know, so
                    # they add no information about `_p` regardless of which
                    # states they touch in the structural graph. Listing them
                    # here would only pollute FAIL-reason strings.
                    u_passthrough_names = {
                        u_name for _y_idx, u_name in self._u_passthrough_meas
                    }
                    true_meas_names = [
                        y for y in self.experiment_list[0].y_names
                        if y not in u_passthrough_names
                    ]
                else:
                    true_meas_names = None
                results = check_structural_identifiability(self.model, p_est_names, true_meas_names)
                self._identifiability_results = results
                warn_unidentifiable(results)
            except ImportError:
                self._identifiability_results = None
            except Exception as e:
                warnings.warn(
                    f"Structural identifiability check failed: {e}. Proceeding with estimation anyway.",
                    UserWarning,
                )
                self._identifiability_results = None
            # --- End identifiability check ---

            # Build Collocation NLP
            self._prepare_nlp()
            self._create_nlp(method="collocation")
            self.flags["collocation_setup"] = True
            
            # Sync flags
            self.flags["setup"] = True


    def set_initial_guess(self):
            """Seed the NLP variables from the best information available.
    
            States are seeded by forward-simulating the model from each
            experiment's initial state at the current parameter guess
            (``_p_est0``); if the simulation fails or produces non-finite
            values, the constant initial state is used instead. Inputs are
            seeded from the experiment's interpolated input data. An explicit
            ``x_guess`` / ``u_guess`` / ``z_guess`` on the experiment overrides
            the corresponding default.
            """
            from do_mpc.estimator._utils import simulate_trajectory

            assert self.flags["setup"], "Backend was not setup yet."
    
            plugin = "idas" if self.model.n_z > 0 else "cvodes"
            integrator_cache = {}
            try:
                p_full = np.array(self._p_cat_fun(self._p_est0, self.p_fun(0))).flatten()
            except Exception:
                p_full = None
    
            for i, experiment in enumerate(self.experiment_list):
                x0_cat = self._convert2struct(experiment.initial_state, self._x).cat
                if experiment.x_guess is not None:
                    self._opt_x_num[f"_x_exp_{i}"] = self._physical_to_optimizer(
                        experiment.x_guess, self._x_scaling, self._x_offset
                    )
                else:
                    n_fe = experiment.n_horizon
                    x_traj = None
                    if p_full is not None:
                        x_traj = simulate_trajectory(
                            self.model, experiment.fe_time,
                            experiment.u[:n_fe], experiment.tvp[:n_fe],
                            p_full, np.array(x0_cat).flatten(),
                            integrator_cache, plugin,
                            {"abstol": 1e-8, "reltol": 1e-8},
                        )
                    if x_traj is not None:
                        for k in range(n_fe + 1):
                            self._opt_x_num[f"_x_exp_{i}", k] = (
                                self._physical_to_optimizer(
                                    x_traj[k].reshape(-1, 1),
                                    self._x_scaling.cat,
                                    self._x_offset.cat,
                                )
                            )
                    else:
                        self._opt_x_num[f"_x_exp_{i}"] = self._physical_to_optimizer(
                            x0_cat, self._x_scaling, self._x_offset
                        )
                if self._z.size > 0:
                    if experiment.z_guess is not None:
                        self._opt_x_num[f"_z_exp_{i}"] = self._physical_to_optimizer(
                            experiment.z_guess, self._z_scaling, self._z_offset
                        )
                    else:
                        self._opt_x_num[f"_z_exp_{i}"] = (
                            self._physical_to_optimizer(
                                self._convert2struct(experiment.initial_algebraic, self._z).cat,
                                self._z_scaling,
                                self._z_offset,
                            )
                        )
                if experiment.u_guess is not None:
                    self._opt_x_num[f"_u_exp_{i}"] = (
                        self._physical_to_optimizer(
                            self._convert2struct(experiment.u_guess, self._u).cat,
                            self._u_scaling,
                            self._u_offset,
                        )
                    )
                elif self._u.size > 0:
                    for k in range(experiment.n_horizon):
                        self._opt_x_num[f"_u_exp_{i}", k] = (
                            self._physical_to_optimizer(
                                experiment.u[k].reshape(-1, 1),
                                self._u_scaling.cat,
                                self._u_offset.cat,
                            )
                        )
    
            self._opt_x_num["_p_est"] = self._p_est_to_optimizer(
                self._p_est0.cat, label="Initial guess (p_est0)"
            )
    
            self.flags["set_initial_guess"] = True

    def estimate_parameters(
        self,
        method: str = "collocation",
        auto_save: bool = True,
        print_identifiability_summary: bool = True,
    ) -> Dict[str, float]:
        """Run parameter estimation and return the estimated parameter values.

        Args:
            method: Estimation method. ``'collocation'`` (the default) discretizes
                the ODE via orthogonal collocation and solves one large NLP with
                IPOPT. ``'single_shooting'`` keeps only the parameters (and,
                optionally, each experiment's initial state) as variables and
                rolls a differentiable CasADi integrator forward, fitting a direct
                weighted least-squares objective with IPOPT.
            auto_save: If True (default), write ``estimation_results.json`` and
                the autosave sidecar in cwd.
            print_identifiability_summary: If True (default), emit the
                post-estimation structural identifiability table here.

        Returns:
            Estimated parameter values keyed by parameter name.
        """
        assert self.flags["setup"], "Estimator was not setup yet. Please call estimator.setup()."

        # Bounds may have changed since setup; the initial guess is validated
        # in set_initial_guess below when it is mapped to optimizer space.
        self._validate_transform_config()

        self._p_est0_seed_snapshot = {
            key: self._slot_value(self._p_est0, key)
            for key in self._p_est.keys() if key != "default"
        }

        # Sync _p_est0 into the NLP variable vector so the optimizer always
        # starts from the latest initial-guess values.
        self.set_initial_guess()

        # Clear any x0 recovered by a previous run; only an x0-estimating backend
        # repopulates this below.
        self.estimated_initial_states = None

        if method == "collocation":
            if not self.flags["collocation_setup"]:
                self.setup()
            for i, experiment in enumerate(self.experiment_list):
                self._opt_p_num[f"_x_prev_exp_{i}"] = experiment.initial_state

                sym_y_meas = self.model.sv.sym_struct(
                    [castools.entry("y_meas", repeat=experiment.n_meas, struct=self._y_meas)]
                )(0)
                sym_y_mask = self.model.sv.sym_struct(
                    [castools.entry("y_mask", repeat=experiment.n_meas, struct=self._y_weight)]
                )(0)
                sym_tvp = self.model.sv.sym_struct(
                    [castools.entry("tvp", repeat=experiment.n_horizon, struct=self._tvp)]
                )(0)
                for k in range(experiment.n_horizon):
                    sym_tvp["tvp", k] = experiment.tvp[k]

                for m in range(experiment.n_meas):
                    sym_y_meas["y_meas", m] = experiment.y_meas[m]
                    sym_y_mask["y_mask", m] = experiment.y_mask[m]

                self._opt_p_num[f"_y_meas_exp_{i}"] = sym_y_meas["y_meas"]
                self._opt_p_num[f"_y_mask_exp_{i}"] = sym_y_mask["y_mask"]
                self._opt_p_num[f"_tvp_exp_{i}"] = sym_tvp["tvp"]

            if hasattr(self, "p_fun") and getattr(self, "_p_set", None) is not None and self._p_set.size > 0:
                self._opt_p_num["_p_set"] = self.p_fun(0.0)
            self.solve()
            result = self._p_est_from_optimizer(self._opt_x_num["_p_est"])
            
            stats = getattr(self, "solver_stats", {}) or {}
            self._last_solver_outcome = {
                "method": "collocation",
                "status": stats.get("return_status"),
                "iterations": stats.get("iter_count"),
                "success": stats.get("success"),
                "extras": {
                    "t_wall_total": stats.get("t_wall_total"),
                },
            }
            
        elif method == "single_shooting":
            # Lazy creation of the single-shooting backend
            if not self.flags["single_shooting_setup"]:
                self._plugin = "idas" if self.model.n_z > 0 else "cvodes"
                self._integrator_opts = dict(self.settings.integrator_opts)
                self._experiment_data = [self._preprocess_experiment(exp) for exp in self.experiment_list]
                self._prepare_single_shooting_nlp()
                self._create_nlp(method="single_shooting")
                self.flags["single_shooting_setup"] = True

            result = self._solve_single_shooting()

            stats = getattr(self, "solver_stats", {}) or {}
            self._last_solver_outcome = {
                "method": "single_shooting",
                "status": stats.get("return_status"),
                "iterations": stats.get("iter_count"),
                "success": stats.get("success"),
                "extras": {
                    "t_wall_total": stats.get("t_wall_total"),
                },
            }
        else:
            raise ValueError(f"Unknown estimation method '{method}'.")
        
        self._store_estimated_params(result)
        self.flags["initial_run"] = True
        self._update_identifiability_report(print_summary=print_identifiability_summary)

        if auto_save:
            try:
                self.save_results()
            except Exception as exc:
                warnings.warn(f"Auto-save of estimation_results.json failed: {exc}")

            # Canonical sidecar: a second copy of the auto-saved payload at a
            # hidden path, immune to later user-side overwrites of
            # estimation_results.json. Kept as a stable backup of the latest autosaved payload.
            try:
                self.save_results(path=".estimation_results_autosave.json", quiet=True)
            except Exception as exc:
                warnings.warn(f"Sidecar auto-save failed: {exc}")

        return self._estimated_params_dict()

    # ---------------------------------------------------------------
    #  Conversions
    # ---------------------------------------------------------------
    
    def _optimizer_to_physical(self, values, scaling, offset):
        """Map optimizer coordinates to physical coordinates."""
        return values * scaling + offset

    def _physical_to_optimizer(self, values, scaling, offset):
        """Map physical coordinates to optimizer coordinates."""
        return (values - offset) / scaling

    def _refresh_p_transform_metadata(self) -> None:
        """Rebuild concatenated parameter transform metadata from p_est and p_set.

        Covers the affine structs (scaling, offset) and the log mask over the
        full ``_p`` vector. ``_p_set`` entries are always linear with unit
        scaling, so the full mask is the ``_p_est`` mask routed through
        ``_p_cat_fun`` into ``_p`` declaration order.
        """
        self._p_scaling = self.model._p(
            self._p_cat_fun(self._p_est_scaling, self._p_set_scaling)
        )
        self._p_offset = self.model._p(
            self._p_cat_fun(self._p_est_offset, self._p_set_offset)
        )
        mask_est = castools.DM(self._p_est_log_mask.astype(float))
        mask_set = castools.DM.zeros(self.n_p_set, 1)
        self._p_log_mask_full = (
            np.array(self._p_cat_fun(mask_est, mask_set)).flatten() > 0.5
        )

    @staticmethod
    def _from_optimizer_map(values, scaling, offset, log_mask):
        """Optimizer -> physical for a vector with per-element transforms.

        Affine (``v * s + o``) for linear elements, ``s * exp(v)`` for
        log-masked elements. ``scaling`` / ``offset`` / ``log_mask`` are flat
        numeric arrays; ``values`` may be a CasADi symbolic column (SX/MX),
        a DM, or a numpy array. Struct views (e.g. ``opt_x['_p_est']``) are
        normalized via ``.cat``. Symbolic inputs are handled per element via
        vertsplit so no dead ``exp`` branch is ever built for linear entries.
        """
        if hasattr(values, "cat"):
            values = values.cat
        s = np.asarray(scaling, dtype=float).reshape(-1)
        o = np.asarray(offset, dtype=float).reshape(-1)
        m = np.asarray(log_mask, dtype=bool).reshape(-1)
        if not m.any():
            if isinstance(values, (castools.SX, castools.MX)):
                return values * s.reshape(-1, 1) + o.reshape(-1, 1)
            arr = np.asarray(values, dtype=float).reshape(-1)
            return (arr * s + o).reshape(-1, 1)
        if isinstance(values, (castools.SX, castools.MX)):
            elems = ca.vertsplit(values)
            out = [
                s[i] * ca.exp(e) if m[i] else e * s[i] + o[i]
                for i, e in enumerate(elems)
            ]
            return ca.vertcat(*out)
        arr = np.asarray(values, dtype=float).reshape(-1)
        out = arr * s + o
        out[m] = s[m] * np.exp(arr[m])
        return out.reshape(-1, 1)

    def _to_optimizer_map(self, values, scaling, offset, log_mask, element_names, label):
        """Physical -> optimizer for a vector with per-element transforms.

        Numeric only (the physical -> optimizer direction is never built
        symbolically). Log-masked elements map as ``log(v / s)`` and must be
        strictly positive; violations raise :class:`ValueError` naming each
        offending element via ``element_names`` and the ``label`` of the
        quantity being mapped (initial guess, bound).
        """
        if hasattr(values, "cat"):
            values = values.cat
        s = np.asarray(scaling, dtype=float).reshape(-1)
        o = np.asarray(offset, dtype=float).reshape(-1)
        m = np.asarray(log_mask, dtype=bool).reshape(-1)
        arr = np.asarray(values, dtype=float).reshape(-1)
        out = (arr - o) / s
        if m.any():
            bad = m & ~(arr > 0)
            if bad.any():
                offenders = [
                    f"  {element_names[i]} = {arr[i]!r}"
                    for i in np.flatnonzero(bad)
                ]
                raise ValueError(
                    f"{label} for log-scale parameters must be strictly "
                    f"positive (they are optimized as z = log(p / scaling)):\n"
                    + "\n".join(offenders)
                )
            out[m] = np.log(arr[m] / s[m])
        return out.reshape(-1, 1)

    def _p_est_from_optimizer(self, values):
        """Optimizer -> physical map for the ``_p_est`` vector (symbolic or numeric)."""
        return self._from_optimizer_map(
            values,
            np.array(self._p_est_scaling.cat).flatten(),
            np.array(self._p_est_offset.cat).flatten(),
            self._p_est_log_mask,
        )

    def _p_est_to_optimizer(self, values, label="value"):
        """Physical -> optimizer map for the ``_p_est`` vector (numeric only)."""
        return self._to_optimizer_map(
            values,
            np.array(self._p_est_scaling.cat).flatten(),
            np.array(self._p_est_offset.cat).flatten(),
            self._p_est_log_mask,
            self._p_est_element_names,
            label,
        )

    def _p_from_optimizer_full(self, values):
        """Optimizer -> physical map over the concatenated ``_p`` vector.

        Requires :meth:`_refresh_p_transform_metadata` to have run so the
        full-vector scaling / offset / log mask are current.
        """
        return self._from_optimizer_map(
            values,
            np.array(self._p_scaling.cat).flatten(),
            np.array(self._p_offset.cat).flatten(),
            self._p_log_mask_full,
        )
    
    
    def set_objective(
            self,
            stage_cost_process: Union[castools.SX, castools.MX],
            stage_cost_measurement: Union[castools.SX, castools.MX],
            arrival_cost: Union[castools.SX, castools.MX],
        ) -> None:
            """Set the objective function for the parameter estimation problem.
    
            Args:
                stage_cost_process: Expression for the process stage cost (depends on w, tvp, p).
                stage_cost_measurement: Expression for the measurement stage cost (depends on v, p).
                arrival_cost: Expression for the arrival cost (depends on x, x_prev).
            """
            assert stage_cost_measurement.shape == (1, 1), (
                "stage_cost_measurement must have shape=(1,1). You have {}".format(
                    stage_cost_measurement.shape
                )
            )
            assert stage_cost_process.shape == (1, 1), (
                "stage_cost_process must have shape=(1,1). You have {}".format(stage_cost_process.shape)
            )
            assert arrival_cost.shape == (1, 1), (
                "arrival_cost must have shape=(1,1). You have {}".format(arrival_cost.shape)
            )
            assert self.flags["setup"] == False, "Cannot call .set_objective after .setup."
            self._P_v_default = None
    
            # Replace model symbolic variables self.model._p with the new variables
            stage_cost_measurement = castools.substitute(
                stage_cost_measurement,
                self._p_est,
                castools.vertcat(*[self.model._p[name] for name in self._p_est.keys()]).reshape(
                    (-1, 1)
                ),
            )
            stage_cost_measurement = castools.substitute(
                stage_cost_measurement,
                self._p_set,
                castools.vertcat(*[self.model._p[name] for name in self._p_set.keys()]).reshape(
                    (-1, 1)
                ),
            )
            stage_cost_process = castools.substitute(
                stage_cost_process,
                self._p_est,
                castools.vertcat(*[self.model._p[name] for name in self._p_est.keys()]).reshape(
                    (-1, 1)
                ),
            )
            stage_cost_process = castools.substitute(
                stage_cost_process,
                self._p_set,
                castools.vertcat(*[self.model._p[name] for name in self._p_set.keys()]).reshape(
                    (-1, 1)
                ),
            )
    
            stage_cost_measurement_input = self._v, self._p
            stage_cost_process_input = self._w, self._tvp, self._p
    
            self.stage_cost_measurement_fun = castools.Function(
                "stage_cost_measurement_fun",
                [*stage_cost_measurement_input],
                [stage_cost_measurement],
            )
            self.stage_cost_process_fun = castools.Function(
                "stage_cost_process_fun", [*stage_cost_process_input], [stage_cost_process]
            )
    
            try:
                self.stage_cost_measurement_fun(
                    *[input_i(0) for input_i in stage_cost_measurement_input]
                )
            except:
                err_msg = "Stage cost equation must be solely depending on v and p. A dependency on tvp is not implemented yet."
                raise Exception(err_msg)
    
            try:
                self.stage_cost_process_fun(*[input_i(0) for input_i in stage_cost_process_input])
            except:
                err_msg = "Stage cost equation must be solely depending on w, p and tvp."
                raise Exception(err_msg)
    
            arrival_cost_input = self._x, self._x_prev
            self.arrival_cost_fun = castools.Function(
                "arrival_cost_fun", [*arrival_cost_input], [arrival_cost]
            )
    
            try:
                self.arrival_cost_fun(*[input_i(0) for input_i in arrival_cost_input])
            except:
                err_msg = "Arrival cost equation must be solely depending on x_0, x_prev, p"
                raise Exception(err_msg)
    
            self.flags["set_objective"] = True
    
    def set_default_objective(
        self,
        P_x: Optional[Union[np.ndarray, castools.SX, castools.MX]] = None,
        P_v: Optional[Union[np.ndarray, castools.SX, castools.MX, castools.DM]] = None,
        P_w: Optional[Union[np.ndarray, castools.SX, castools.MX, castools.DM]] = None,
        *,
        sigma_x: Optional[Dict[str, float]] = None,
        sigma_y: Optional[Dict[str, float]] = None,
        sigma_w: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set the default weighted least-squares objective.

        Two equivalent forms:

        * **Dict form (preferred).** Pass per-channel std-devs keyed by name.
            The estimator builds ``diag(1/sigma**2)`` so callers do not have to
            worry about state / measurement ordering. Each dict must cover the
            model's full keyset for that quantity exactly — unknown or missing
            keys raise ``ValueError``.

            * ``sigma_x``: keyed by state name (``model._x``).
            * ``sigma_y``: keyed by measurement name registered with
            ``meas_noise=True`` (i.e. the names that produced a ``_v`` slot).
            * ``sigma_w``: keyed by state name declared with
            ``process_noise=True`` in ``set_rhs`` (i.e. the names that
            produced a ``_w`` slot).

        * **Matrix form (power user).** Pass ``P_x`` / ``P_v`` / ``P_w`` as
            full ``(n_x, n_x)`` / ``(n_v, n_v)`` / ``(n_w, n_w)`` matrices when
            off-diagonal weighting is genuinely required.

        Mutual exclusion is per quantity: ``P_x`` xor ``sigma_x``, ``P_v`` xor
        ``sigma_y``, ``P_w`` xor ``sigma_w``. Mixing matrix and dict forms
        across the three quantities (e.g. matrix ``P_x`` + dict ``sigma_y``)
        is fine.

        If neither ``P_x`` nor ``sigma_x`` is given, ``P_x`` defaults to
        ``np.eye(n_x)`` with a ``UserWarning``. ``P_v`` and ``P_w`` have no
        soft fallback — when the model has noise variables you must supply
        the corresponding weights via either form.
        """
        n_x = self.model.n_x
        n_v = self.model.n_v
        n_w = self.model.n_w

        if P_x is not None and sigma_x is not None:
            raise TypeError("Pass either P_x or sigma_x for the arrival cost, not both.")
        if P_v is not None and sigma_y is not None:
            raise TypeError("Pass either P_v or sigma_y for measurement noise, not both.")
        if P_w is not None and sigma_w is not None:
            raise TypeError("Pass either P_w or sigma_w for process noise, not both.")

        if sigma_x is not None:
            P_x = self._diag_from_sigma_dict(sigma_x, self.model._x, "sigma_x")
        elif P_x is None:
            warnings.warn(
                "set_default_objective called without P_x or sigma_x; falling back to "
                "P_x = np.eye(n_x). Specify sigma_x={state_name: sigma_value, ...} to "
                "weight each state's arrival cost by its initial-state uncertainty.",
                UserWarning,
                stacklevel=2,
            )
            P_x = np.eye(n_x)

        if sigma_y is not None:
            # _v slot names are `<meas_name>_noise` (do-mpc convention from
            # set_meas(name, expr, meas_noise=True)); strip the suffix so the
            # dict can be keyed by the public measurement name.
            P_v = self._diag_from_sigma_dict(
                sigma_y, self.model._v, "sigma_y", strip_suffix="_noise"
            )

        if sigma_w is not None:
            # _w slot names are `<state_name>_noise` (do-mpc convention from
            # set_rhs(name, expr, process_noise=True)); same suffix strip.
            P_w = self._diag_from_sigma_dict(
                sigma_w, self.model._w, "sigma_w", strip_suffix="_noise"
            )

        input_types = (np.ndarray, castools.SX, castools.MX, castools.DM)
        err_msg = "{name} must be of type {type_set}, you have {type_is}"
        assert isinstance(P_x, input_types), err_msg.format(
            name="P_x", type_set=input_types, type_is=type(P_x)
        )
        input_types_optional = (np.ndarray, castools.SX, castools.MX, castools.DM, type(None))
        assert isinstance(P_v, input_types_optional), err_msg.format(
            name="P_v", type_set=input_types_optional, type_is=type(P_v)
        )
        assert isinstance(P_w, input_types_optional), err_msg.format(
            name="P_w", type_set=input_types_optional, type_is=type(P_w)
        )

        assert P_x.shape == (n_x, n_x), "P_x has wrong shape:{}, must be {}".format(
            P_x.shape, (n_x, n_x)
        )

        # Calculate stage cost
        stage_cost_measurement = castools.DM(0)
        p_v_numeric = None

        if P_v is None:
            assert n_v == 0, (
                "Must pass weighting factor P_v (or sigma_y), since you have "
                "measurement noise on some measurements (configured in model)."
            )
        else:
            assert P_v.shape == (n_v, n_v), "P_v has wrong shape:{}, must be {}".format(
                P_v.shape, (n_v, n_v)
            )
            try:
                p_v_numeric = np.array(P_v, dtype=float)
            except Exception:
                p_v_numeric = None
                warnings.warn(
                    "Could not convert P_v to numeric array for identifiability reporting. "
                    "Identity fallback will be used when needed.",
                    UserWarning,
                )
            v = self._v.cat
            stage_cost_measurement += v.T @ P_v @ v

        stage_cost_process = castools.DM(0)

        if P_w is None:
            assert n_w == 0, (
                "Must pass weighting factor P_w (or sigma_w), since you have "
                "process noise on some states (configured in model)."
            )
        else:
            assert P_w.shape == (n_w, n_w), "P_w has wrong shape:{}, must be {}".format(
                P_w.shape, (n_w, n_w)
            )
            w = self._w.cat
            stage_cost_process += w.T @ P_w @ w

        # Calculate arrival cost
        x_0 = self._x
        x_prev = self._x_prev
        dx = x_0.cat - x_prev.cat
        arrival_cost = dx.T @ P_x @ dx

        # Set estimator objective
        self.set_objective(stage_cost_process, stage_cost_measurement, arrival_cost)
        self._P_v_default = p_v_numeric
    
    @staticmethod
    def _diag_from_sigma_dict(
        sigma: Dict[str, float],
        target_struct,
        label: str,
        strip_suffix: str = "",
    ) -> np.ndarray:
        """Build ``diag(1/sigma**2)`` from a name-keyed dict.

        ``target_struct`` is a do-mpc CasADi struct (``model._x``, ``model._v``,
        ``model._w``) whose ``.keys()`` define the canonical ordering. When
        ``strip_suffix`` is set, each slot key has that suffix removed before
        lookup in ``sigma`` — used to translate ``_v`` / ``_w`` slot names
        (``<base_name>_noise``) back to the public base name.
        """
        if not isinstance(sigma, dict):
            raise TypeError(
                f"{label} must be a dict mapping name -> sigma value; got {type(sigma).__name__}."
            )

        expected_names: List[str] = []
        for slot in target_struct.keys():
            if slot == "default":
                continue
            name = slot[: -len(strip_suffix)] if strip_suffix and slot.endswith(strip_suffix) else slot
            expected_names.append(name)

        provided = set(sigma.keys())
        expected = set(expected_names)
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing keys {missing}")
            if extra:
                parts.append(f"unknown keys {extra}")
            raise ValueError(
                f"{label} keys do not match the model surface ({'; '.join(parts)}). "
                f"Expected exactly: {sorted(expected)}."
            )

        values = np.array([float(sigma[name]) for name in expected_names], dtype=float)
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            bad = {n: sigma[n] for n in expected_names if not (np.isfinite(sigma[n]) and sigma[n] > 0)}
            raise ValueError(
                f"{label} values must be finite and strictly positive; got {bad}."
            )
        return np.diag(1.0 / values**2)

    # ---------------------------------------------------------------
    #  Parameter estimation (dispatch to backend)
    # ---------------------------------------------------------------
    def _store_estimated_params(self, result: np.ndarray) -> None:
        """Update ``_p_est0`` with the values returned by the backend.

        Both collocation and single-shooting backends return a numpy array whose
        length equals ``self.n_p_est``: a flat concatenation of every
        ``_p_est`` slot in declaration order. Writing it through
        ``_p_est0.master`` preserves per-element shape for vector slots;
        per-slot scalar assignment would broadcast a single scalar across
        every element of any vector slot.

        Asserts each slot's values land inside its user-set bounds after
        writeback (with a small numerical tolerance for IPOPT's
        ``acceptable_tol`` slack). A gross violation indicates either a
        backend / writeback misattribution or that the bounds were never
        propagated to IPOPT — both have happened historically, so this is
        a permanent guardrail.
        """
        flat = np.asarray(result, dtype=float).reshape(-1, 1)
        if flat.shape[0] != self.n_p_est:
            raise ValueError(
                f"Backend returned {flat.shape[0]} values, but _p_est holds "
                f"{self.n_p_est} elements across {len([k for k in self._p_est.keys() if k != 'default'])} slot(s)."
            )
        self._p_est0.master = castools.DM(flat)

        violations: List[str] = []
        for key in self._p_est.keys():
            if key == "default":
                continue
            vals = np.asarray(self._p_est0[key]).reshape(-1)
            lb = np.asarray(self._p_est_lb[key]).reshape(-1)
            ub = np.asarray(self._p_est_ub[key]).reshape(-1)
            # Tolerance: generous enough to absorb IPOPT's bound-slack
            # (typically <= acceptable_tol of 1e-6 .. 1e-3 in scaled units,
            # which transforms back to a comparable physical magnitude), but
            # tight enough that an order-of-magnitude misattribution trips.
            for i, (v, l, u) in enumerate(zip(vals, lb, ub)):
                scale = max(1.0, abs(l) if np.isfinite(l) else 0.0,
                            abs(u) if np.isfinite(u) else 0.0)
                tol = 1e-4 * scale + 1e-6
                if np.isfinite(l) and v < l - tol:
                    violations.append(
                        f"  {key}[{i}] = {v!r} < lower bound {l!r}"
                    )
                if np.isfinite(u) and v > u + tol:
                    violations.append(
                        f"  {key}[{i}] = {v!r} > upper bound {u!r}"
                    )
        if violations:
            raise RuntimeError(
                "Estimated parameters fall outside their user-set bounds "
                "after writeback. This usually indicates a backend writeback "
                "misattribution (the wrong slice of the flat result was "
                "stored in this slot) or that the bounds were not propagated "
                "to the NLP. Violations:\n" + "\n".join(violations)
            )

    @staticmethod
    def _slot_value(struct, key) -> Union[float, List[float]]:
        """Return a struct slot as a JSON-friendly scalar or list.

        Scalar slots (size 1) yield a Python ``float``; vector slots yield
        a list of ``float`` so the per-element values survive JSON
        serialisation. Used everywhere the estimator builds a
        ``{name: value}`` mapping for downstream consumers (auto-save,
        seed snapshot, fit-report payload).
        """
        arr = np.asarray(struct[key]).reshape(-1)
        if arr.size == 1:
            return float(arr[0])
        return [float(v) for v in arr]

    def _estimated_params_dict(self) -> Dict[str, Union[float, List[float]]]:
        """Return the current estimated parameters as a name-keyed dict.

        Scalar parameters map to ``float``; vector parameters map to a
        ``list[float]`` of length ``n``.
        """
        p_est_keys = [k for k in self._p_est.keys() if k != "default"]
        return {key: self._slot_value(self._p_est0, key) for key in p_est_keys}


    def _build_save_payload(self) -> Dict[str, Any]:
        """Build the JSON payload written by :py:meth:`save_results`.

        Single source of truth for the on-disk schema.
        """
        estimated = self._estimated_params_dict()

        fixed = {}
        if hasattr(self, "p_fun") and self._p_set.size > 0:
            try:
                p_set_vals = self.p_fun(0)
                for key in self._p_set.keys():
                    if key == "default":
                        continue
                    fixed[key] = self._slot_value(p_set_vals, key)
            except Exception:
                pass

        # Identifiability payload: structural reachability only.
        ident: dict = {}
        results = getattr(self, "_identifiability_results", None)
        if results:
            ident["structural"] = {
                r.param_name: {
                    "identifiable": bool(r.identifiable),
                    "reason": r.reason,
                }
                for r in results
            }

        return {
            "estimated_parameters": estimated,
            "fixed_parameters": fixed,
            "identifiability": ident,
            "p_est0_seed": getattr(self, "_p_est0_seed_snapshot", {}),
            "solver": getattr(self, "_last_solver_outcome", None),
            # Optimizer-space transforms per parameter; values are physical
            # regardless.
            "transforms": dict(self._p_est_transform),
        }

    def save_results(
        self,
        path: Union[str, Path] = "estimation_results.json",
        quiet: bool = False,
    ) -> Path:
        """Save estimated and fixed parameters plus identifiability metrics.

        Automatically called after :py:meth:`estimate_parameters`.  Can also
        be called manually to write results to a custom path.

        Args:
            path: Output file path (default: ``estimation_results.json`` in cwd).
            quiet: If True, suppress the "saved to ..." print.

        Returns:
            The resolved path that was written.
        """
        path = Path(path)
        payload = self._build_save_payload()
        path.write_text(json.dumps(payload, indent=2) + "\n")
        if not quiet:
            print(f"Estimation results saved to {path}")
        return path

    # ---------------------------------------------------------------
    #  Forward simulation
    # ---------------------------------------------------------------
    def simulate_experiments(
        self,
        experiments: Optional[List] = None,
    ) -> List[SimulationResult]:
        """Forward-simulate experiments and compute predicted measurements.

        Uses the current estimated + fixed parameter values to integrate the
        model ODE forward in time for each experiment, then evaluates the
        measurement equation at each measurement time to produce predicted
        outputs.

        This is backend-agnostic and works after either collocation or single-shooting
        estimation.  It can also be called with fresh :class:`Experiment`
        objects (e.g. validation or test data) that were **not** part of the
        estimation.

        Args:
            experiments: List of :class:`Experiment` instances to simulate.
                If ``None``, defaults to ``self.experiment_list`` (the training
                experiments used for estimation).

        Returns:
            List of :class:`SimulationResult`, one per experiment.
        """
        if experiments is None:
            experiments = self.experiment_list
        if not experiments:
            raise ValueError("No experiments to simulate.")

        model = self.model
        p_full = np.array(self._p_cat_fun(self._p_est0, self.p_fun(0))).flatten()
        # Same integrator configuration as the single-shooting backend, so
        # the simulation that is scored is the simulation that was fitted.
        plugin = self.settings.integrator_plugin
        if plugin == "auto":
            plugin = "idas" if model.n_z > 0 else "cvodes"
        intg_opts = dict(self.settings.integrator_opts)
        integrator_cache: Dict[str, object] = {}

        results = []
        for exp in experiments:
            n_fe = len(exp.fe_time) - 1
            x0_raw = exp.initial_state
            x0 = np.array(x0_raw.cat if hasattr(x0_raw, "cat") else x0_raw).flatten()

            # Integrate over the union of the FE grid and the measurement
            # times so predictions are read at exactly the times the data
            # was sampled, not at the nearest FE boundary.
            t_grid, owner_interval, _ = build_refined_grid(
                exp.fe_time, exp.meas_time
            )
            u_seq = exp.u[:n_fe][owner_interval]
            tvp_seq = exp.tvp[:n_fe][owner_interval]

            x_traj = simulate_trajectory(
                model, t_grid, u_seq, tvp_seq,
                p_full, x0, integrator_cache, plugin, intg_opts,
                raise_on_failure=True,
            )

            y_pred = evaluate_measurements(
                model, x_traj, t_grid, exp.meas_time,
                u_seq, tvp_seq, p_full,
            )

            results.append(SimulationResult(
                time=exp.meas_time.copy(),
                y_pred=y_pred,
                y_meas=exp.y_meas.copy(),
                y_mask=exp.y_mask.copy(),
                meas_names=list(exp.all_meas_names),
            ))

        return results

    @staticmethod
    def compute_rmse(predicted: np.ndarray, measured: np.ndarray) -> float:
        """Root mean square error, ignoring NaN pairs."""
        mask = ~(np.isnan(predicted) | np.isnan(measured))
        if not np.any(mask):
            return float("nan")
        return float(np.sqrt(np.mean((predicted[mask] - measured[mask]) ** 2)))

    @classmethod
    def compute_nrmse(cls, predicted: np.ndarray, measured: np.ndarray) -> float:
        """Range-normalized RMSE, ignoring NaN pairs."""
        mask = ~(np.isnan(predicted) | np.isnan(measured))
        if not np.any(mask):
            return float("nan")
        measured_valid = measured[mask]
        data_range = float(np.max(measured_valid) - np.min(measured_valid))
        if data_range <= 1e-10:
            return (
                0.0
                if np.allclose(predicted[mask], measured_valid)
                else float("inf")
            )
        rmse = cls.compute_rmse(predicted[mask], measured_valid)
        return float(rmse / data_range)

    def evaluate_fit(
        self,
        sim_results: List[SimulationResult],
        *,
        include_measurements: Optional[List[str]] = None,
        exclude_inputs: bool = True,
    ) -> FitReport:
        """Compute canonical RMSE / NRMSE metrics from simulation results.

        Args:
            sim_results: Results returned by :meth:`simulate_experiments`.
            include_measurements: Optional explicit subset of measurement
                names to score. If ``None``, all channels are considered.
            exclude_inputs: If ``True`` (default), channels corresponding to
                external ``_u`` inputs are excluded from scoring.

        Returns:
            Structured :class:`FitReport` with per-experiment channel metrics
            and a mean-NRMSE convenience property.
        """
        include_set = set(include_measurements) if include_measurements is not None else None
        excluded_inputs = (
            {k for k in self.model.u.keys() if k != "default"} if exclude_inputs else set()
        )

        per_experiment: List[ExperimentFitReport] = []
        for exp_idx, sim_res in enumerate(sim_results):
            channel_metrics: Dict[str, ChannelFitMetrics] = {}
            for ch_idx, ch_name in enumerate(sim_res.meas_names):
                if include_set is not None and ch_name not in include_set:
                    continue
                if ch_name in excluded_inputs:
                    continue

                mask = sim_res.y_mask[:, ch_idx].astype(bool)
                n_valid = int(np.count_nonzero(mask))
                if n_valid == 0:
                    continue

                pred = sim_res.y_pred[mask, ch_idx]
                meas = sim_res.y_meas[mask, ch_idx]
                channel_metrics[ch_name] = ChannelFitMetrics(
                    rmse=self.compute_rmse(pred, meas),
                    nrmse=self.compute_nrmse(pred, meas),
                    n_valid=n_valid,
                )

            per_experiment.append(
                ExperimentFitReport(
                    experiment_index=exp_idx,
                    channel_metrics=channel_metrics,
                )
            )

        return FitReport(per_experiment=per_experiment)

    def simulate_and_evaluate_fit(
        self,
        experiments: Optional[List] = None,
        *,
        include_measurements: Optional[List[str]] = None,
        exclude_inputs: bool = True,
        save_results: bool = False,
    ) -> Tuple[List[SimulationResult], FitReport]:
        """Convenience wrapper around simulation plus canonical fit scoring.

        Args:
            experiments: Optional list of experiments to simulate. If None,
                the training experiments added via ``add_experiment`` are used
                (this is the pre-fit-check path).
            include_measurements: Optional explicit subset of measurement
                names to score. Forwarded to :meth:`evaluate_fit`.
            exclude_inputs: When True (default) channels corresponding to
                external ``_u`` inputs are excluded from scoring.
            save_results: When True, write a sibling ``validation_results.json``
                file alongside ``estimation_results.json`` (in the current
                working directory). The post-estimation validation call is
                the primary use case; pre-fit calls may opt in too, in which
                case the same filename is overwritten.

        Returns:
            Tuple of ``(sim_results, fit_report)``.
        """
        sim_results = self.simulate_experiments(experiments)
        fit_report = self.evaluate_fit(
            sim_results,
            include_measurements=include_measurements,
            exclude_inputs=exclude_inputs,
        )
        if save_results:
            try:
                self._save_fit_results(
                    fit_report,
                    experiments=experiments,
                    include_measurements=include_measurements,
                    exclude_inputs=exclude_inputs,
                )
            except Exception as exc:
                warnings.warn(
                    f"save_results=True: failed to write validation_results.json: {exc}"
                )
        return sim_results, fit_report

    def _build_fit_results_payload(
        self,
        fit_report: FitReport,
        *,
        experiments: Optional[List],
        include_measurements: Optional[List[str]],
        exclude_inputs: bool,
    ) -> Dict[str, Any]:
        """Build the JSON payload written by :meth:`_save_fit_results`.

        Captures the per-experiment per-channel NRMSE / RMSE table the
        printed fit-report contains, the parameter values used
        for the simulation, the experiment identifiers, and a timestamp so
        the file is self-describing alongside ``estimation_results.json``.
        """
        from datetime import datetime, timezone

        per_experiment_payload: List[Dict[str, Any]] = []
        for exp_idx, exp_report in enumerate(fit_report.per_experiment):
            channel_payload: Dict[str, Dict[str, Any]] = {}
            for ch_name, metrics in exp_report.channel_metrics.items():
                channel_payload[ch_name] = {
                    "rmse": (
                        float(metrics.rmse) if np.isfinite(metrics.rmse) else None
                    ),
                    "nrmse": (
                        float(metrics.nrmse) if np.isfinite(metrics.nrmse) else None
                    ),
                    "n_valid": int(metrics.n_valid),
                }
            per_experiment_payload.append(
                {
                    "experiment_index": int(exp_report.experiment_index),
                    "experiment_id": self._experiment_identifier(
                        experiments, exp_idx
                    ),
                    "channels": channel_payload,
                }
            )

        mean_nrmse = fit_report.mean_nrmse
        mean_value = (
            float(mean_nrmse)
            if mean_nrmse is not None and np.isfinite(mean_nrmse)
            else None
        )

        # Parameter values currently held by the estimator. These reflect
        # whatever was solved for (post-estimation) or seeded (pre-fit).
        estimated = self._estimated_params_dict()
        fixed: Dict[str, Union[float, List[float]]] = {}
        if hasattr(self, "p_fun") and self._p_set.size > 0:
            try:
                p_set_vals = self.p_fun(0)
                for key in self._p_set.keys():
                    if key == "default":
                        continue
                    fixed[key] = self._slot_value(p_set_vals, key)
            except Exception:
                pass

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiments": [
                self._experiment_identifier(experiments, idx)
                for idx in range(len(fit_report.per_experiment))
            ],
            "settings": {
                "include_measurements": (
                    list(include_measurements)
                    if include_measurements is not None
                    else None
                ),
                "exclude_inputs": bool(exclude_inputs),
            },
            "estimated_parameters": estimated,
            "fixed_parameters": fixed,
            "mean_nrmse": mean_value,
            "per_experiment": per_experiment_payload,
        }

    def _experiment_identifier(
        self, experiments: Optional[List], idx: int
    ) -> Optional[str]:
        """Best-effort identifier for one experiment in the simulated list.

        The Experiment class does not require a stable name, so this prefers
        explicit ``name`` / ``label`` attributes, falls back to a CSV path
        when one is hung off the experiment, and finally returns ``None``.
        """
        exp_list = experiments if experiments is not None else self.experiment_list
        if not exp_list or idx >= len(exp_list):
            return None
        exp = exp_list[idx]
        for attr in ("name", "label", "identifier"):
            value = getattr(exp, attr, None)
            if value:
                return str(value)
        for attr in ("csv_path", "data_path", "source_path", "path"):
            value = getattr(exp, attr, None)
            if value:
                return str(value)
        return None

    def _save_fit_results(
        self,
        fit_report: FitReport,
        *,
        experiments: Optional[List],
        include_measurements: Optional[List[str]],
        exclude_inputs: bool,
        path: Union[str, Path] = "validation_results.json",
    ) -> Path:
        """Serialize a fit report to JSON.

        The default filename is ``validation_results.json`` because the
        primary use case is the post-estimation validation call. Pre-fit
        calls that opt in via ``save_results=True`` write to the same name
        and overwrite. Pass a different path when both payloads must be retained.
        """
        out_path = Path(path)
        payload = self._build_fit_results_payload(
            fit_report,
            experiments=experiments,
            include_measurements=include_measurements,
            exclude_inputs=exclude_inputs,
        )
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        return out_path

    # ---------------------------------------------------------------
    #  Plotting
    # ---------------------------------------------------------------
    def plot_results(
        self,
        meas_names: List[str] = None,
        experiment_index: Union[int, List[int]] = None,
        show_fe_times: bool = True,
    ) -> None:
        """Plot the results of the parameter estimation.

        Delegates to :py:func:`do_mpc.parameter_estimator._plotting.plot_results`.

        Args:
            meas_names: List of measurement names to plot. If None, all measurements are plotted.
            experiment_index: Index or list of indices of experiments to plot. If None, all experiments are plotted.
            show_fe_times: If True, indications for finite element time points are added to the plots.
        """
        return _plot_results_fn(self, meas_names, experiment_index, show_fe_times)

    # ---------------------------------------------------------------
    #  Validation
    # ---------------------------------------------------------------

    def _check_validity(self) -> None:
            """Private method called during setup. Checks if the configuration is valid."""
            # Objective must be defined.
            if self.flags["set_objective"] == False:
                raise Exception(
                    "Objective is undefined. Please call .set_objective() or .set_default_objective() prior to .setup()."
                )
    
            # p_fun must be set, if p are defined in model.
            if self.flags["set_p_fun"] == False and self._p_set.size > 0:
                raise Exception(
                    "You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() prior to setup."
                )
    
            # Lower bounds should be lower than upper bounds
            for lb, ub in zip(
                [self._x_lb, self._u_lb, self._z_lb], [self._x_ub, self._u_ub, self._z_ub]
            ):
                bound_check = lb.cat > ub.cat
                bound_fail = [label_i for i, label_i in enumerate(lb.labels()) if bound_check[i]]
                if np.any(bound_check):
                    raise Exception(
                        "Your bounds are inconsistent. For {} you have lower bound > upper bound.".format(
                            bound_fail
                        )
                    )
    
            # Set dummy functions for tvp and p in case these parameters are unused.
            if "p_fun" not in self.__dict__:
                _p = self.get_p_template()
    
                def p_fun(t):
                    return _p
    
                self.set_p_fun(p_fun)
    
    # ---------------------------------------------------------------
    #  Bounds
    # ---------------------------------------------------------------
    @do_mpc.tools.IndexedProperty
    def bounds(self, ind):
        """Query and set bounds of the optimization variables.
        The :py:func:`bounds` method is an indexed property, meaning
        getting and setting this property requires an index and calls this function.
        The power index (elements are separated by commas) must contain at least the following elements:

        ======      =================   ==========================================================
        order       index name          valid options
        ======      =================   ==========================================================
        1           bound type          ``lower`` and ``upper``
        2           variable type       ``_x``, ``_u`` and ``_z`` (and ``_p_est`` for MHE)
        3           variable name       Names defined in :py:class:`do_mpc.model.Model`.
        ======      =================   ==========================================================

        **Example**:

        ::

            optimizer.bounds['lower','_x', 'phi_1'] = -2*np.pi
            optimizer.bounds['upper','_x', 'phi_1'] = 2*np.pi
        """
        assert isinstance(ind, tuple), (
            "Power index must include bound_type, var_type, var_name (as a tuple)."
        )
        assert len(ind) >= 3, (
            "Power index must include bound_type, var_type, var_name (as a tuple)."
        )
        bound_type = ind[0]
        var_type = ind[1]
        var_name = ind[2:]

        err_msg = "Invalid power index {} for bound_type. Must be from (lower, upper)."
        assert bound_type in ("lower", "upper"), err_msg.format(bound_type)
        err_msg = "Invalid power index {} for var_type. Must be from (_x, _u, _z, _p_est)."
        assert var_type in ("_x", "_u", "_z", "_p_est"), err_msg.format(var_type)

        if bound_type == "lower":
            query = "{var_type}_{bound_type}".format(var_type=var_type, bound_type="lb")
        elif bound_type == "upper":
            query = "{var_type}_{bound_type}".format(var_type=var_type, bound_type="ub")

        var_struct = getattr(self, query)

        err_msg = "Calling .bounds with {} is not valid. Possible keys are {}."
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), (
            err_msg.format(ind, var_struct.keys())
        )

        return var_struct[var_name]

    @bounds.setter
    def bounds(self, ind, val):
        """See Docstring for bounds getter method"""
        assert isinstance(ind, tuple), (
            "Power index must include bound_type, var_type, var_name (as a tuple)."
        )
        assert len(ind) >= 3, (
            "Power index must include bound_type, var_type, var_name (as a tuple)."
        )
        bound_type = ind[0]
        var_type = ind[1]
        var_name = ind[2:]

        err_msg = "Invalid power index {} for bound_type. Must be from (lower, upper)."
        assert bound_type in ("lower", "upper"), err_msg.format(bound_type)
        err_msg = "Invalid power index {} for var_type. Must be from (_x, _u, _z, _p_est)."
        assert var_type in ("_x", "_u", "_z", "_p_est"), err_msg.format(var_type)

        if bound_type == "lower":
            query = "{var_type}_{bound_type}".format(var_type=var_type, bound_type="lb")
        elif bound_type == "upper":
            query = "{var_type}_{bound_type}".format(var_type=var_type, bound_type="ub")

        var_struct = getattr(self, query)

        err_msg = "Calling .bounds with {} is not valid. Possible keys are {}."
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), (
            err_msg.format(ind, var_struct.keys())
        )

        var_struct[var_name] = val

        # Update bounds of optimization variables, if the problem is already created
        if self.flags.get("prepare_nlp", False):
            self._update_bounds()

    # ---------------------------------------------------------------
    #  Scaling
    # ---------------------------------------------------------------
    @do_mpc.tools.IndexedProperty
    def scaling(self, ind):
        """Query and set scaling of the optimization variables.

        **Example**:

        ::

            optimizer.scaling['_x', 'phi_1'] = 2
            optimizer.scaling['_x', 'phi_2'] = 2
        """
        assert isinstance(ind, tuple), "Power index must include var_type, var_name (as a tuple)."
        assert len(ind) >= 2, "Power index must include var_type, var_name (as a tuple)."
        var_type = ind[0]
        var_name = ind[1:]

        err_msg = "Invalid power index {} for var_type. Must be from (_x, _u, _z, _p_est)."
        assert var_type in ("_x", "_u", "_z", "_p_est"), err_msg.format(var_type)

        query = "{var_type}_scaling".format(var_type=var_type)
        var_struct = getattr(self, query)

        err_msg = "Calling .scaling with {} is not valid. Possible keys are {}."
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), (
            err_msg.format(ind, var_struct.keys())
        )

        return var_struct[var_name]

    @scaling.setter
    def scaling(self, ind, val):
        """See Docstring for scaling getter method"""
        assert not self.flags["setup"], (
            "Scaling can only be set before the optimization problem is created."
        )
        assert isinstance(ind, tuple), "Power index must include var_type, var_name (as a tuple)."
        assert len(ind) >= 2, "Power index must include var_type, var_name (as a tuple)."
        var_type = ind[0]
        var_name = ind[1:]

        err_msg = "Invalid power index {} for var_type. Must be from (_x, _u, _z, _p_est)."
        assert var_type in ("_x", "_u", "_z", "_p_est"), err_msg.format(var_type)

        query = "{var_type}_scaling".format(var_type=var_type)
        var_struct = getattr(self, query)

        err_msg = "Calling .scaling with {} is not valid. Possible keys are {}."
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), (
            err_msg.format(ind, var_struct.keys())
        )

        var_struct[var_name] = val

    # ---------------------------------------------------------------
    #  Transform
    # ---------------------------------------------------------------
    @do_mpc.tools.IndexedProperty
    def transform(self, ind):
        """Query and set the optimizer-space transform of an estimated parameter.

        ``'linear'`` (the default) optimizes the parameter through the affine
        scaling/offset map. ``'log'`` optimizes ``z = log(p / scaling)``
        instead, so a parameter whose plausible range spans decades becomes a
        well-scaled additive quantity for the solver. Everything user-facing
        (bounds, ``p_est0``, results, saved JSON) stays in physical space.

        A log-scale parameter requires a strictly positive, finite bounds
        bracket and a strictly positive initial guess. For vector parameters
        the transform applies to every element of the slot.

        **Example**:

        ::

            estimator.transform['_p_est', 'A'] = 'log'

        """
        assert isinstance(ind, tuple), "Power index must include var_type, var_name (as a tuple)."
        assert len(ind) >= 2, "Power index must include var_type, var_name (as a tuple)."
        var_type = ind[0]
        var_name = ind[1] if isinstance(ind[1], str) else ind[1][0]

        err_msg = "Invalid power index {} for var_type. Transforms apply to estimated parameters (_p_est)."
        assert var_type == "_p_est", err_msg.format(var_type)
        err_msg = "Calling .transform with {} is not valid. Possible keys are {}."
        assert var_name in self._p_est.keys(), err_msg.format(ind, self._p_est.keys())

        return self._p_est_transform.get(var_name, "linear")

    @transform.setter
    def transform(self, ind, val):
        """See docstring for transform getter method."""
        assert not self.flags["setup"], (
            "Transforms can only be set before the optimization problem is created."
        )
        assert isinstance(ind, tuple), "Power index must include var_type, var_name (as a tuple)."
        assert len(ind) >= 2, "Power index must include var_type, var_name (as a tuple)."
        var_type = ind[0]
        var_name = ind[1] if isinstance(ind[1], str) else ind[1][0]

        err_msg = "Invalid power index {} for var_type. Transforms apply to estimated parameters (_p_est)."
        assert var_type == "_p_est", err_msg.format(var_type)
        err_msg = "Calling .transform with {} is not valid. Possible keys are {}."
        assert var_name in self._p_est.keys(), err_msg.format(ind, self._p_est.keys())
        assert val in ("linear", "log"), (
            f"Invalid transform {val!r}. Must be 'linear' or 'log'."
        )

        idx = self._p_est.f[var_name]
        if val == "log":
            self._p_est_transform[var_name] = "log"
            self._p_est_log_mask[idx] = True
        else:
            self._p_est_transform.pop(var_name, None)
            self._p_est_log_mask[idx] = False

    # ---------------------------------------------------------------
    #  Non-linear constraints
    # ---------------------------------------------------------------
    def set_nl_cons(
        self,
        expr_name: str,
        expr: Union[castools.SX, castools.MX],
        ub: float = np.inf,
        soft_constraint: bool = False,
        penalty_term_cons: int = 1,
        maximum_violation: float = np.inf,
    ) -> Union[castools.SX, castools.MX]:
        """Introduce new constraint to the class.

        Expressions must be formulated with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        They are implemented as:

        .. math::

            m(x,u,z,p_{\\text{tv}}, p) \\leq m_{\\text{ub}}

        Args:
            expr_name: Arbitrary name for the given expression.
            expr: CasADi SX or MX function.
            ub: Upper bound
            soft_constraint: Flag to enable soft constraint
            penalty_term_cons: Penalty term constant
            maximum_violation: Maximum violation

        Returns:
            Returns the newly created expression.
        """
        assert self.flags["setup"] == False, "Cannot call .set_expression after .setup()."
        assert isinstance(expr_name, str), "expr_name must be str, you have: {}".format(
            type(expr_name)
        )
        assert isinstance(expr, (castools.SX, castools.MX)), (
            "expr must be a casadi SX or MX type, you have: {}".format(type(expr))
        )
        assert isinstance(ub, (int, float, np.ndarray)), (
            "ub must be float, int or numpy.ndarray, you have: {}".format(type(ub))
        )
        assert isinstance(soft_constraint, bool), (
            "soft_constraint must be boolean, you have: {}".format(type(soft_constraint))
        )

        if soft_constraint == True:
            self.slack_vars_list.extend(
                [
                    {
                        "slack_name": expr_name,
                        "shape": expr.shape,
                        "ub": maximum_violation,
                        "penalty": penalty_term_cons,
                    }
                ]
            )

        self.nl_cons_list.extend([{"expr_name": expr_name, "expr": expr, "ub": ub}])

        return expr

    def _setup_nl_cons(self, nl_cons_input: Union[castools.SX, castools.MX]) -> None:
        """Private method called from backend setup.
        Creates the non-linear constraint structs and functions.

        Args:
            nl_cons_input: list of symbolic variables used as input to the nl_cons function.
        """
        self._eps = _eps = self.model.sv.sym_struct(
            [
                castools.entry(slack_i["slack_name"], shape=slack_i["shape"])
                for slack_i in self.slack_vars_list
            ]
        )
        self._nl_cons = self.model.sv.struct(
            [
                castools.entry(expr_i["expr_name"], expr=expr_i["expr"])
                for expr_i in self.nl_cons_list
            ]
        )

        self.n_eps = _eps.shape[0]
        self._eps_lb = _eps(0.0)
        self._eps_ub = _eps(np.inf)

        for slack_i in self.slack_vars_list:
            self._eps_ub[slack_i["slack_name"]] = slack_i["ub"]
            self._nl_cons[slack_i["slack_name"]] -= self._eps[slack_i["slack_name"]]
            self.slack_cost += castools.sum1(slack_i["penalty"] * self._eps[slack_i["slack_name"]])

        self.epsterm_fun = castools.Function("epsterm", [_eps], [self.slack_cost])

        nl_cons_input += [_eps]
        self._nl_cons_fun = castools.Function("nl_cons_fun", nl_cons_input, [self._nl_cons])

        self._nl_cons_ub = self._nl_cons(np.inf)
        self._nl_cons_lb = self._nl_cons(-np.inf)
        for nl_cons_i in self.nl_cons_list:
            self._nl_cons_ub[nl_cons_i["expr_name"]] = nl_cons_i["ub"]

    # ---------------------------------------------------------------
    #  Template methods
    # ---------------------------------------------------------------
    def get_y_template(
        self, experiment
    ) -> Union[castools.structure3.SXStruct, castools.structure3.MXStruct]:
        """Obtain output template for measurement function.

        Returns:
            y_template
        """
        y_template = self.model.sv.sym_struct(
            [castools.entry("y_meas", repeat=experiment.n_horizon + 1, struct=self._y_meas)]
        )
        return y_template(0)

    def get_tvp_template(
        self, experiment
    ) -> Union[castools.structure3.SXStruct, castools.structure3.MXStruct]:
        """Obtain output template for time-varying parameter function.

        Returns:
            Casadi SX or MX structure
        """
        tvp_template = self.model.sv.sym_struct(
            [castools.entry("_tvp", repeat=experiment.n_horizon + 1, struct=self.model._tvp)]
        )
        return tvp_template(0)

    def get_p_template(self) -> _TrackedPTemplate:
        """Obtain output template for the (not estimated) parameter function.

        Returns:
            p_template
        """
        return _make_tracked_p_template(self._p_set)

    def set_p_fun(
        self,
        p_fun: Callable[[float], _TrackedPTemplate],
    ) -> None:
        """Set function which returns the (fixed) parameters.

        Args:
            p_fun: Parameter function. Must return the tracked template
                produced by :py:meth:`get_p_template`, with every fixed
                parameter explicitly assigned.
        """
        template = p_fun(0)
        if not isinstance(template, _TrackedPTemplate):
            raise ValueError(
                "set_p_fun: p_fun must return the template produced by "
                "get_p_template(). Build it with "
                "`template = estimator.get_p_template()`, assign every "
                "fixed parameter via `template['<name>'] = <value>`, and "
                "return it from p_fun."
            )
        if self.get_p_template().labels() != template.labels():
            raise ValueError(
                "set_p_fun: p_fun returned a struct with a different shape "
                "than get_p_template(). Use the template produced by "
                "estimator.get_p_template() unmodified."
            )

        required = {k for k in self._p_set.keys() if k != "default"}
        missing = sorted(required - template._set_keys)
        if missing:
            raise ValueError(
                "set_p_fun: the following non-estimated parameters were "
                f"never assigned a value (still at the default 0.0): {missing}.\n"
                "  Set each one via `p_template['<name>'] = <value>` before "
                "calling set_p_fun(). If zero is intentional, assign 0.0 "
                "explicitly to record the choice. Refer to the model "
                "documentation for the expected values."
            )

        self.p_fun = p_fun
        self.flags["set_p_fun"] = True

    def get_identifiability_report(self, P_v: Optional[np.ndarray] = None):
        """Return the latest structural identifiability report.

        Only structural identifiability is reported. ``P_v`` is accepted for
        backward compatibility and is ignored when provided.
        """
        if P_v is not None:
            warnings.warn(
                "P_v is ignored: only structural identifiability is reported.",
                UserWarning,
                stacklevel=2,
            )

        results = getattr(self, "_identifiability_results", None)
        if not results:
            self._last_identifiability_report = None
            return None

        report = {
            "structural": {
                r.param_name: {
                    "identifiable": bool(r.identifiable),
                    "directly_affects_states": list(r.directly_affects_states),
                    "reachable_states": list(r.reachable_states),
                    "reachable_measured_states": list(r.reachable_measured_states),
                    "reason": r.reason,
                }
                for r in results
            }
        }
        self._last_identifiability_report = report
        return report

    @property
    def last_identifiability_report(self):
        """Return the cached structural identifiability report, if available."""
        return self._last_identifiability_report

    def _update_identifiability_report(
        self,
        print_summary: bool = True,
        p_est_values: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print and cache the structural identifiability summary after estimation."""
        from do_mpc.estimator._identifiability import (
            print_structural_identifiability_summary,
        )

        results = getattr(self, "_identifiability_results", None)
        if not results:
            self._last_identifiability_report = None
            return
        self.get_identifiability_report()

        if print_summary:
            if p_est_values is None:
                try:
                    p_est_values = self._estimated_params_dict()
                except Exception:
                    p_est_values = None
            try:
                print_structural_identifiability_summary(
                    results, p_est_values
                )
            except Exception as exc:
                warnings.warn(
                    f"Structural identifiability summary failed to print: {exc}",
                    UserWarning,
                )

    # ---------------------------------------------------------------
    #  Utility
    # ---------------------------------------------------------------
    def _convert2struct(self, val, struct):
        """Convert a value to match a CasADi struct format."""
        if val is None:
            return struct(0)
        elif isinstance(val, (int, float)):
            return struct(val)
        elif isinstance(val, np.ndarray):
            return struct(val)
        elif isinstance(val, (castools.DM, castools.structure3.DMStruct)):
            return struct(val)
        else:
            raise ValueError(f"Cannot convert value of type {type(val)} to struct.")


