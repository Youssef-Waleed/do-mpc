"""Structural identifiability checks for the parameter estimator.

Provides a graph-based reachability analysis to detect parameters that
cannot be identified from the available measurements, regardless of data
quality. This catches cases where a parameter only affects states that
have no path (through the ODE coupling) to any measured output.

This module is intentionally self-contained and loosely coupled to the
rest of the parameter estimator. It only depends on a do_mpc Model and
can be removed without affecting estimation functionality.
"""

import warnings
import numpy as np
import casadi
import casadi.tools as castools
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class IdentifiabilityResult:
    """Result of a structural identifiability check for one parameter.

    The ``*_states`` lists name both differential states (``_x``) and
    algebraic states (``_z``); the tracer walks the full DAE incidence
    graph rather than only the ODE jacobian.
    """

    param_name: str
    identifiable: bool
    directly_affects_states: List[str]
    reachable_states: List[str]
    reachable_measured_states: List[str]
    reason: str = ""


def _sparsity_to_bool(jac_expr) -> np.ndarray:
    """Convert a CasADi Jacobian expression's sparsity pattern to a boolean matrix."""
    nrow, ncol = jac_expr.shape
    mat = np.zeros((nrow, ncol), dtype=bool)
    sp = jac_expr.sparsity()
    rows = sp.row()
    cols = sp.get_col()
    for k in range(sp.nnz()):
        mat[rows[k], cols[k]] = True
    return mat


def _transitive_closure(adj: np.ndarray) -> np.ndarray:
    """Compute the transitive closure of a boolean adjacency matrix.

    Uses repeated squaring (Floyd-Warshall-like) to find all reachable nodes.
    adj[i, j] = True means there is a direct edge from j to i
    (i.e., state i depends on state j in the RHS).

    Returns a matrix where result[i, j] = True means state i is reachable
    from state j through the coupling graph.
    """
    n = adj.shape[0]
    reach = adj.copy()
    # Include self-loops (a state can reach itself)
    np.fill_diagonal(reach, True)
    # Iterate until convergence
    for _ in range(n):
        new_reach = reach @ reach > 0  # matrix multiply, then booleanize
        if np.array_equal(new_reach, reach):
            break
        reach = new_reach
    return reach


def check_structural_identifiability(
    model,
    p_est_names: List[str],
    true_meas_names: Optional[List[str]] = None,
) -> List[IdentifiabilityResult]:
    """Check whether each estimated parameter is structurally identifiable.

    Uses the sparsity structure of the model's Jacobians to determine
    whether each parameter has a causal path to at least one measurement:

        parameter -> (affects) -> state(s) -> (couples to) -> ... -> measured state

    This is a necessary condition for identifiability. If a parameter has
    no such path, it cannot be estimated from the available measurements
    regardless of the amount or quality of data.

    Args:
        model: A configured and setup do_mpc.model.Model instance.
        p_est_names: List of parameter names that will be estimated.
        true_meas_names: List of measurement names that are true system
            outputs (not input pass-throughs). If None, all measurements
            are considered true outputs. This distinction is important
            because input measurements (e.g., measured actuator signals)
            do not provide information about the system's internal states.

    Returns:
        List of IdentifiabilityResult, one per estimated parameter.
    """
    assert model.flags["setup"], "Model must be setup before checking identifiability."

    # --- Get variable names (skip 'default' entries in CasADi structs) ---
    x_names = [k for k in model._x.keys() if k != "default"]
    z_names = [k for k in model._z.keys() if k != "default"]
    p_names = [k for k in model._p.keys() if k != "default"]
    y_names = [k for k in model._y.keys() if k != "default"]
    n_x = len(x_names)
    n_z = len(z_names)
    n_nodes = n_x + n_z

    # Unified node ordering: differential states first, then algebraic.
    node_names = x_names + z_names

    # If true_meas_names not specified, use all measurements
    if true_meas_names is None:
        true_meas_names = y_names

    # Validate that requested parameter names exist
    for p_name in p_est_names:
        if p_name not in p_names:
            raise ValueError(
                f"Parameter '{p_name}' not found in model. Available: {p_names}"
            )

    # --- Compute Jacobian sparsity patterns ---
    # d(rhs)/d(x): RHS of differential state i depends on differential state j
    jac_rhs_x = casadi.jacobian(model._rhs.cat, castools.vertcat(model._x))
    sp_rhs_x = _sparsity_to_bool(jac_rhs_x)
    # d(rhs)/d(z): RHS of differential state i depends on algebraic state j
    if n_z > 0:
        jac_rhs_z = casadi.jacobian(model._rhs.cat, castools.vertcat(model._z))
        sp_rhs_z = _sparsity_to_bool(jac_rhs_z)
    else:
        sp_rhs_z = np.zeros((n_x, 0), dtype=bool)
    # d(rhs)/d(p): parameter j enters the RHS of differential state i
    jac_rhs_p = casadi.jacobian(model._rhs.cat, castools.vertcat(model._p))
    sp_rhs_p = _sparsity_to_bool(jac_rhs_p)
    # d(alg)/d(x|z|p): same for the algebraic residual block
    if n_z > 0:
        jac_alg_x = casadi.jacobian(model._alg.cat, castools.vertcat(model._x))
        sp_alg_x = _sparsity_to_bool(jac_alg_x)
        jac_alg_z = casadi.jacobian(model._alg.cat, castools.vertcat(model._z))
        sp_alg_z = _sparsity_to_bool(jac_alg_z)
        jac_alg_p = casadi.jacobian(model._alg.cat, castools.vertcat(model._p))
        sp_alg_p = _sparsity_to_bool(jac_alg_p)
    else:
        sp_alg_x = np.zeros((0, n_x), dtype=bool)
        sp_alg_z = np.zeros((0, 0), dtype=bool)
        sp_alg_p = np.zeros((0, len(p_names)), dtype=bool)
    # d(y)/d(x), d(y)/d(z): which states / algebraics appear in each measurement
    jac_y_x = casadi.jacobian(model._y_expression.cat, castools.vertcat(model._x))
    sp_y_x = _sparsity_to_bool(jac_y_x)
    if n_z > 0:
        jac_y_z = casadi.jacobian(model._y_expression.cat, castools.vertcat(model._z))
        sp_y_z = _sparsity_to_bool(jac_y_z)
    else:
        sp_y_z = np.zeros((len(y_names), 0), dtype=bool)

    # --- Build index mappings ---
    p_idx = {name: i for i, name in enumerate(p_names)}
    node_idx = {name: i for i, name in enumerate(node_names)}

    # Boolean mask: which measurements are "true" (not input pass-throughs)
    true_meas_mask = np.array([name in true_meas_names for name in y_names])

    # --- Build the DAE incidence graph on the combined node set N = _x ∪ _z.
    # adj[i, j] = True means "influence can flow from node j to node i".
    #   - Differential coupling is directional: jac(rhs_i, var_j) means var_j
    #     enters the dynamics of x_i, so the edge points var_j -> x_i.
    #   - Algebraic coupling is bidirectional. An algebraic residual g_k = 0
    #     implicitly relates every variable it touches, so the standard DAE
    #     incidence-graph treatment is to add edges in both directions among
    #     all variables in supp(g_k). This is conservative: it may credit
    #     identifiability when the residual is rank-deficient, but it never
    #     misses a real path (consistent with the "necessary, not sufficient"
    #     guarantee of structural identifiability).
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    # x_j -> x_i (ODE)
    adj[:n_x, :n_x] |= sp_rhs_x
    # z_j -> x_i (ODE references algebraic)
    if n_z > 0:
        adj[:n_x, n_x:] |= sp_rhs_z
        # Algebraic incidence: for each residual row, mark every variable pair
        # in its support with edges in both directions.
        alg_support = np.concatenate([sp_alg_x, sp_alg_z], axis=1)  # (n_z, n_nodes)
        for k in range(n_z):
            supp = np.where(alg_support[k])[0]
            if supp.size:
                ii, jj = np.meshgrid(supp, supp, indexing="ij")
                adj[ii, jj] = True

    reach = _transitive_closure(adj)

    # --- For each estimated parameter, check reachability to measurements ---
    results = []
    for p_name in p_est_names:
        pi = p_idx[p_name]

        # Nodes the parameter directly enters: _x rows of jac(rhs, p) and
        # _z rows of jac(alg, p). Differential and algebraic equations are
        # treated symmetrically; a parameter that only appears in set_alg
        # is still a direct influence on the constrained algebraic state(s).
        direct_nodes = [x_names[i] for i in range(n_x) if sp_rhs_p[i, pi]]
        if n_z > 0:
            direct_nodes += [z_names[i] for i in range(n_z) if sp_alg_p[i, pi]]

        # All nodes reachable from those direct nodes via the DAE graph.
        reachable: set = set()
        for n_name in direct_nodes:
            ni = node_idx[n_name]
            for j in range(n_nodes):
                if reach[j, ni]:
                    reachable.add(node_names[j])

        # Which reachable nodes appear in a true measurement?
        measured_reachable = []
        for n_name in reachable:
            ni = node_idx[n_name]
            if ni < n_x:
                col = ni
                sp_col = sp_y_x[:, col]
            else:
                col = ni - n_x
                sp_col = sp_y_z[:, col]
            for yi, y_name in enumerate(y_names):
                if true_meas_mask[yi] and sp_col[yi]:
                    measured_reachable.append(y_name)

        measured_reachable = list(set(measured_reachable))
        is_identifiable = len(measured_reachable) > 0

        # Build reason string
        if is_identifiable:
            reason = ""
        else:
            if not direct_nodes:
                reason = (
                    f"Parameter '{p_name}' does not appear in any state's dynamics "
                    "or algebraic residual."
                )
            else:
                reason = (
                    f"Parameter '{p_name}' affects states {direct_nodes}, "
                    f"reachable states are {sorted(reachable)}, but none of these "
                    f"appear in any true measurement ({sorted(true_meas_names)})."
                )

        results.append(
            IdentifiabilityResult(
                param_name=p_name,
                identifiable=is_identifiable,
                directly_affects_states=direct_nodes,
                reachable_states=sorted(reachable),
                reachable_measured_states=sorted(measured_reachable),
                reason=reason,
            )
        )

    return results


def print_structural_identifiability_summary(
    results: List[IdentifiabilityResult],
    p_est_values: Optional[Dict[str, float]] = None,
) -> None:
    """Print a compact post-estimation summary of structural identifiability.

    Complements the estimator output with a structural reachability check.
    Recovery error versus the documented value is the trusted signal; this
    table only confirms whether each estimated parameter has *any* causal
    path to a measurement. Parameters flagged FAIL here cannot be identified
    from the available measurement set regardless of how much data is fed in.

    Args:
        results: Output of ``check_structural_identifiability``.
        p_est_values: Optional ``{param_name: estimated_value}`` mapping. When
            provided, the value column is filled in alongside the OK / FAIL
            verdict. When ``None`` only the verdict column is shown. Values
            are displayed as provided by the caller.
    """
    if not results:
        return

    print("\nIdentifiability report (structural):")
    if p_est_values is not None:
        print("  param                value        struct")
    else:
        print("  param                              struct")

    for r in results:
        status = "OK" if r.identifiable else "FAIL"
        if p_est_values is not None and r.param_name in p_est_values:
            value = float(p_est_values[r.param_name])
            print(f"  {r.param_name:<20} {value:>11.4g}  {status:>7}")
        else:
            print(f"  {r.param_name:<32} {status:>7}")
        if not r.identifiable and r.reason:
            print(f"    reason: {r.reason}")


def warn_unidentifiable(results: List[IdentifiabilityResult]) -> List[str]:
    """Emit warnings for any unidentifiable parameters.

    Args:
        results: Output from check_structural_identifiability.

    Returns:
        List of warning message strings (for programmatic use).
    """
    messages = []
    for r in results:
        if not r.identifiable:
            msg = (
                f"Structural identifiability warning: {r.reason} "
                f"Estimation of '{r.param_name}' will likely produce unreliable results."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            messages.append(msg)
    return messages
