import numpy as np
import casadi as ca
import casadi.tools as castools
from dataclasses import asdict

# Import utilities needed for Single Shooting
from do_mpc.estimator._utils import build_dae, build_refined_grid

class BatchOptimizer:
    """Base class handling parallel multi-experiment NLP construction."""
    
    def __init__(self, model):
        self.model = model
        self.experiment_list = []
        
        # --- Shared NLP Structures ---
        self.S = None
        self.solver_stats = None
        self._nlp_obj = None
        self.flags = {
            "setup": False,
            "initial_run": False,
            "set_initial_guess": False,
        }

        # --- Collocation Specific Structures ---
        self._opt_x = None
        self._opt_x_num = None
        self.opt_x_num_unscaled = None
        self._opt_p = None
        self._opt_p_num = None
        self._opt_aux = None
        self.opt_aux_num = None
        self._nlp_cons = None
        self._nlp_cons_lb = None
        self._nlp_cons_ub = None
        self._lb_opt_x = None
        self._ub_opt_x = None
        self.opt_x_scaling = None
        self.opt_x_offset = None
        self.opt_x_log_mask = None
        self._p_est_opt_x_range = None
        self.discretization_list = None
        self.lam_g_num = None
        self.lam_x_num = None
        
        # --- Single Shooting Specific Structures ---
        self._plugin = "cvodes"
        self._integrator_opts = {}
        self._integrator_cache = {}
        self._experiment_data = []
        self._lbx = None
        self._ubx = None


    # ---------------------------------------------------------------
    #  Experiment management
    # ---------------------------------------------------------------
    def add_experiment(self, experiment) -> None:
            """Add an experiment to the estimator.
    
            Args:
                experiment: A :py:class:`do_mpc.parameter_estimator.Experiment` instance.
            """
            self.experiment_list.append(experiment)

    
    # ---------------------------------------------------------------
    #  Core NLP construction
    # ---------------------------------------------------------------
    def _prepare_nlp(self):
            """Internal method. Builds optimization variable/parameter structs,
            sets up discretization, assembles objective and constraints."""
    
            self.settings.check_for_mandatory_settings()
            nl_cons_input = self.model["x", "u", "z", "tvp"]
            nl_cons_input += [self._p_est, self._p_set]
            self._setup_nl_cons(nl_cons_input)
            
            self._check_validity()
    
            self._refresh_p_transform_metadata()
    
            # Set up discretization for each experiment
            self.discretization_list = [
                self._setup_discretization(experiment) for experiment in self.experiment_list
            ]
    
            struct_list = []
            for i, (
                experiment,
                (ifcn_k, n_total_coll_points, collocation_functions),
            ) in enumerate(zip(self.experiment_list, self.discretization_list)):
                if self.settings.nl_cons_single_slack:
                    n_eps = 1
                else:
                    n_eps = experiment.n_horizon
    
                struct_list += [
                    castools.entry(
                        f"_x_exp_{i}",
                        repeat=[experiment.n_horizon + 1, 1 + n_total_coll_points],
                        struct=self.model._x,
                    ),
                    castools.entry(
                        f"_z_exp_{i}",
                        repeat=[experiment.n_horizon, max(n_total_coll_points, 1)],
                        struct=self.model._z,
                    ),
                    castools.entry(f"_u_exp_{i}", repeat=[experiment.n_horizon], struct=self.model._u),
                    castools.entry(f"_w_exp_{i}", repeat=[experiment.n_horizon], struct=self.model._w),
                    castools.entry(f"_v_exp_{i}", repeat=[experiment.n_meas], struct=self.model._v),
                    castools.entry(f"_eps_exp_{i}", repeat=[n_eps], struct=self._eps),
                ]
    
            # Parameter estimates are defined only once for all experiments.
            struct_list += [
                castools.entry("_p_est", struct=self._p_est),
            ]
    
            self._opt_x = opt_x = self.model.sv.sym_struct(struct_list)
            self.n_opt_x = opt_x.shape[0]
    
            # Create affine structs for optimizer-space -> physical-space mapping.
            self.opt_x_scaling = opt_x_scaling = opt_x(1)
            self.opt_x_offset = opt_x_offset = opt_x(0)
            for i in range(len(self.experiment_list)):
                opt_x_scaling[f"_x_exp_{i}"] = self._x_scaling
                opt_x_offset[f"_x_exp_{i}"] = self._x_offset
                opt_x_scaling[f"_z_exp_{i}"] = self._z_scaling
                opt_x_offset[f"_z_exp_{i}"] = self._z_offset
                opt_x_scaling[f"_u_exp_{i}"] = self._u_scaling
                opt_x_offset[f"_u_exp_{i}"] = self._u_offset
    
            opt_x_scaling["_p_est"] = self._p_est_scaling
            opt_x_offset["_p_est"] = self._p_est_offset
    
            # Flat index range of the _p_est block inside opt_x: struct entries
            # occupy contiguous flat ranges, so the per-element transform can be
            # spliced over a single slice.
            p_idx = np.asarray(opt_x.f["_p_est"]).reshape(-1).astype(int)
            p_lo, p_hi = int(p_idx[0]), int(p_idx[-1]) + 1
            assert np.array_equal(p_idx, np.arange(p_lo, p_hi)), (
                "_p_est block is not contiguous in opt_x."
            )
            self._p_est_opt_x_range = (p_lo, p_hi)
            self.opt_x_log_mask = np.zeros(self.n_opt_x, dtype=bool)
            self.opt_x_log_mask[p_lo:p_hi] = self._p_est_log_mask
    
            unscaled_cat = self._optimizer_to_physical(
                opt_x.cat, opt_x_scaling.cat, opt_x_offset.cat
            )
            if self._p_est_log_mask.any():
                unscaled_cat = castools.vertcat(
                    unscaled_cat[:p_lo],
                    self._p_est_from_optimizer(opt_x.cat[p_lo:p_hi]),
                    unscaled_cat[p_hi:],
                )
            self.opt_x_unscaled = opt_x_unscaled = opt_x(unscaled_cat)
    
            # Build parameter struct for the NLP
            struct_list = []
            for i, experiment in enumerate(self.experiment_list):
                struct_list += [
                    castools.entry(f"_x_prev_exp_{i}", struct=self._x_prev),
                    castools.entry(f"_y_meas_exp_{i}", repeat=[experiment.n_meas], struct=self._y_meas),
                    castools.entry(
                        f"_y_mask_exp_{i}",
                        repeat=[experiment.n_meas],
                        struct=self._y_weight,
                    ),
                    castools.entry(
                        f"_tvp_exp_{i}",
                        repeat=[experiment.n_horizon],
                        struct=self.model._tvp,
                    ),
                ]
    
            struct_list += [
                castools.entry("_p_set", struct=self._p_set),
            ]
    
            self._opt_p = opt_p = self.model.sv.sym_struct(struct_list)
            self.n_opt_p = opt_p.shape[0]
    
            # Auxiliary expressions struct
            struct_list = []
            for i, experiment in enumerate(self.experiment_list):
                struct_list += [
                    castools.entry(
                        f"_aux_exp_{i}",
                        repeat=[experiment.n_horizon],
                        struct=self.model._aux,
                    ),
                ]
    
            self.aux_struct = aux_struct = self.model.sv.sym_struct(struct_list)
            self._opt_aux = opt_aux = self.model.sv.struct(self.aux_struct)
            self.n_opt_aux = opt_aux.shape[0]
    
            self._lb_opt_x = opt_x(-np.inf)
            self._ub_opt_x = opt_x(np.inf)
    
            # Initialize objective function and constraints
            obj = castools.DM(0)
            cons = []
            cons_lb = []
            cons_ub = []
    
            # Concatenate parameters in optimizer space for the discretization
            # layer, then build physical-space views for measurements, objectives
            # and nonlinear constraints.
            _p_opt = self._p_cat_fun(opt_x["_p_est"], opt_p["_p_set"])
            _p = self._p_from_optimizer_full(_p_opt)
            _p_est_phys = self._p_est_from_optimizer(opt_x["_p_est"])
    
            # For all control intervals
            for j, (
                experiment,
                (ifcn_k, n_total_coll_points, collocation_functions),
            ) in enumerate(zip(self.experiment_list, self.discretization_list)):
                for k in range(experiment.n_horizon):
                    # Compute constraints and predicted next state
                    col_xk = castools.vertcat(*opt_x[f"_x_exp_{j}", k + 1, :-1])
                    col_zk = castools.vertcat(*opt_x[f"_z_exp_{j}", k])
                    [g_ksb, xf_ksb] = ifcn_k[k](
                        opt_x[f"_x_exp_{j}", k, -1],
                        col_xk,
                        opt_x[f"_u_exp_{j}", k],
                        col_zk,
                        opt_p[f"_tvp_exp_{j}", k],
                        _p_opt,
                        opt_x[f"_w_exp_{j}", k],
                    )
    
                    # Add the collocation equations
                    cons.append(g_ksb)
                    cons_lb.append(np.zeros(g_ksb.shape[0]))
                    cons_ub.append(np.zeros(g_ksb.shape[0]))
    
                    # Add continuity constraints
                    cons.append(xf_ksb - opt_x[f"_x_exp_{j}", k + 1, -1])
                    cons_lb.append(np.zeros((self.model.n_x, 1)))
                    cons_ub.append(np.zeros((self.model.n_x, 1)))
    
                    if self.settings.nl_cons_single_slack:
                        n_eps = 1
                    else:
                        n_eps = experiment.n_horizon
    
                    k_eps = min(k, n_eps - 1)
                    if self.settings.nl_cons_check_colloc_points:
                        for i in range(n_total_coll_points):
                            nl_cons_k = self._nl_cons_fun(
                                opt_x_unscaled[f"_x_exp_{j}", k, i],
                                opt_x_unscaled[f"_u_exp_{j}", k],
                                opt_x_unscaled[f"_z_exp_{j}", k, i],
                                opt_p[f"_tvp_exp_{j}", k],
                                _p_est_phys,
                                opt_p["_p_set"],
                                opt_x_unscaled[f"_eps_exp_{j}", k_eps],
                            )
                            cons.append(nl_cons_k)
                            cons_lb.append(self._nl_cons_lb)
                            cons_ub.append(self._nl_cons_ub)
                    else:
                        nl_cons_k = self._nl_cons_fun(
                            opt_x_unscaled[f"_x_exp_{j}", k, -1],
                            opt_x_unscaled[f"_u_exp_{j}", k],
                            opt_x_unscaled[f"_z_exp_{j}", k, 0],
                            opt_p[f"_tvp_exp_{j}", k],
                            _p_est_phys,
                            opt_p["_p_set"],
                            opt_x_unscaled[f"_eps_exp_{j}", k_eps],
                        )
                        cons.append(nl_cons_k)
                        cons_lb.append(self._nl_cons_lb)
                        cons_ub.append(self._nl_cons_ub)
    
                    obj += self.stage_cost_process_fun(
                        opt_x_unscaled[f"_w_exp_{j}", k], opt_p[f"_tvp_exp_{j}", k], _p
                    )
    
                    obj += self.epsterm_fun(opt_x_unscaled[f"_eps_exp_{j}", k_eps])
    
                    opt_aux[f"_aux_exp_{j}", k] = self.model._aux_expression_fun(
                        opt_x_unscaled[f"_x_exp_{j}", k, -1],
                        opt_x_unscaled[f"_u_exp_{j}", k],
                        opt_x_unscaled[f"_z_exp_{j}", k, -1],
                        opt_p[f"_tvp_exp_{j}", k],
                        _p,
                    )
    
                # Arrival cost: added ONCE per experiment (outside the FE loop)
                arrival_cost = self.arrival_cost_fun(
                    opt_x_unscaled[f"_x_exp_{j}", 0, -1], opt_p[f"_x_prev_exp_{j}"]
                )
                obj += arrival_cost
    
                if not hasattr(experiment, "meas_time") or experiment.n_meas == 0:
                    raise ValueError(
                        f"Experiment {j} has no measurement times defined. "
                        "Please check the experiment configuration."
                    )
    
                # For each measurement time point
                for m in range(experiment.n_meas):
                    fe_idx = experiment.meas_fe_indices[m]
                    sub_element_idx = experiment.meas_sub_element_indices[m]
                    tau_value = experiment.meas_tau_values[m]
    
                    if fe_idx < 0 or fe_idx >= experiment.n_horizon:
                        continue
    
                    fe_interp_fun = collocation_functions[fe_idx][sub_element_idx]
    
                    x_interp = fe_interp_fun(
                        tau_value,
                        opt_x_unscaled[f"_x_exp_{j}", fe_idx, -1],
                        castools.vertcat(*opt_x_unscaled[f"_x_exp_{j}", fe_idx + 1, :-1]),
                    )
    
                    y_calc = self.model._meas_fun(
                        x_interp,
                        opt_x_unscaled[f"_u_exp_{j}", fe_idx],
                        opt_x_unscaled[f"_z_exp_{j}", fe_idx, 0],
                        opt_p[f"_tvp_exp_{j}", fe_idx],
                        _p,
                        opt_x_unscaled[f"_v_exp_{j}", m],
                    )
    
                    # Drop rows for `_u` pass-through measurements: those channels
                    # are pinned to data via bounds instead of equality constraints
                    # (see _update_bounds + ParameterEstimator._detect_u_passthrough_meas).
                    keep_idx = self._u_passthrough_keep_y_indices
                    if len(keep_idx) == self.model.n_y:
                        residual = (y_calc - opt_p[f"_y_meas_exp_{j}", m]) * opt_p[f"_y_mask_exp_{j}", m]
                        cons.append(residual)
                        cons_lb.append(np.zeros((self.model.n_y, 1)))
                        cons_ub.append(np.zeros((self.model.n_y, 1)))
                    elif keep_idx:
                        residual = (y_calc - opt_p[f"_y_meas_exp_{j}", m]) * opt_p[f"_y_mask_exp_{j}", m]
                        cons.append(residual[keep_idx])
                        cons_lb.append(np.zeros((len(keep_idx), 1)))
                        cons_ub.append(np.zeros((len(keep_idx), 1)))
    
                    obj += self.stage_cost_measurement_fun(opt_x_unscaled[f"_v_exp_{j}", m], _p)
    
            self._update_bounds()
    
            self._nlp_obj = obj
            self._nlp_cons = cons
            self._nlp_cons_lb = cons_lb
            self._nlp_cons_ub = cons_ub
    
            # Initialize numerical structures
            self._opt_x_num = self._opt_x(0)
            self.opt_x_num_unscaled = self._opt_x(0)
            self._opt_p_num = self._opt_p(0)
            self.opt_aux_num = self._opt_aux(0)
    
            self.flags["prepare_nlp"] = True


    def _create_nlp(self, method="collocation"):
            """Internal method. Concatenates constraints, creates IPOPT solver."""

            nlpsol_opts = {
                            "expand": False,
                            "ipopt.linear_solver": "mumps",
                        }
            nlpsol_opts.update(self.settings.nlpsol_opts)

            if method == "collocation":
            
                self._nlp_cons = castools.vertcat(*self._nlp_cons)
                self._nlp_cons_lb = castools.vertcat(*self._nlp_cons_lb)
                self._nlp_cons_ub = castools.vertcat(*self._nlp_cons_ub)

                # Validity check
                _test_obj_fun = castools.Function("f", [self._opt_x, self._opt_p], [self._nlp_obj])
                _test_cons_fun = castools.Function("f", [self._opt_x, self._opt_p], [self._nlp_cons])
                try:
                    _test_obj_fun(self._opt_x_num, self._opt_p_num)
                except:
                    raise Exception(
                        "The optimization problem objective function contains unknown symbolic variables."
                    )
                try:
                    _test_cons_fun(self._opt_x_num, self._opt_p_num)
                except:
                    raise Exception(
                        "The optimization problem constraint function contains unknown symbolic variables."
                    )

                self.n_opt_lagr = self._nlp_cons.shape[0]

                # Create casadi optimization object
                nlpsol_opts = {
                    "expand": False,
                    "ipopt.linear_solver": "mumps",
                }
                nlpsol_opts.update(self.settings.nlpsol_opts)
                self.nlp = {
                    "x": castools.vertcat(self._opt_x),
                    "f": self._nlp_obj,
                    "g": self._nlp_cons,
                    "p": castools.vertcat(self._opt_p),
                }
                self.S = castools.nlpsol("S", "ipopt", self.nlp, nlpsol_opts)

                # Create function to calculate all auxiliary expressions
                self.opt_aux_expression_fun = castools.Function(
                    "opt_aux_expression_fun", [self._opt_x, self._opt_p], [self._opt_aux]
                )

                # Gather meta information
                meta_data = {key: getattr(self.settings, key) for key in asdict(self.settings).keys()}

            elif method == "single_shooting":
                self.nlp = {"x": self._nlp_x, "f": self._nlp_obj} 
                self.S = ca.nlpsol("S", "ipopt", self.nlp, nlpsol_opts)       

    def _setup_discretization(self, experiment):
        """Set up collocation discretization for a single experiment.
        Returns (ifcn_k, n_total_coll_points, collocation_functions)."""
        
        # Scaled variables
        _x, _u, _z, _tvp, _p, _w = self.model["x", "u", "z", "tvp", "p", "w"]
        
        # Unscale variables
        _x_unscaled = self._optimizer_to_physical(_x, self._x_scaling.cat, self._x_offset.cat)
        _u_unscaled = self._optimizer_to_physical(_u, self._u_scaling.cat, self._u_offset.cat)
        _z_unscaled = self._optimizer_to_physical(_z, self._z_scaling.cat, self._z_offset.cat)
        _p_unscaled = self._p_from_optimizer_full(_p)
        
        # Create _rhs and _alg
        _rhs = self.model._rhs_fun(_x_unscaled, _u_unscaled, _z_unscaled, _tvp, _p_unscaled, _w)
        _alg = self.model._alg_fun(_x_unscaled, _u_unscaled, _z_unscaled, _tvp, _p_unscaled, _w)
        
        # Scale (only _rhs)
        _rhs_scaled = _rhs / self._x_scaling.cat
        
        if self.model.model_type == "discrete":
            _i = self.model.sv.sym("i", 0)
            ifcn = castools.Function("ifcn", [_x, _i, _u, _z, _tvp, _p, _w], [_alg, _rhs_scaled])
            n_total_coll_points = 0
        elif self.settings.state_discretization == "collocation":
            ffcn = castools.Function("ffcn", [_x, _u, _z, _tvp, _p, _w], [_rhs_scaled])
            afcn = castools.Function("afcn", [_x, _u, _z, _tvp, _p, _w], [_alg])
        
            coll = experiment.settings.collocation_type
            deg = experiment.settings.collocation_deg
            ni = experiment.settings.collocation_ni
            nk = experiment.n_horizon
            n_x = self.model.n_x
            n_u = self.model.n_u
            n_p = self.model.n_p
            n_z = self.model.n_z
            n_w = self.model.n_w
            n_tvp = self.model.n_tvp
            n_total_coll_points = (deg + 1) * ni
        
            # Choose collocation points
            if coll == "legendre":
                tau_root = [0] + castools.collocation_points(deg, "legendre")
            elif coll == "radau":
                tau_root = [0] + castools.collocation_points(deg, "radau")
            else:
                raise Exception("Unknown collocation scheme")
        
            t_steps = np.diff(experiment.fe_time)
        
            # Coefficients of the collocation equation
            C = np.zeros((deg + 1, deg + 1))
            D = np.zeros(deg + 1)
        
            tau = self.model.sv.sym("tau")
        
            T = np.zeros((nk, ni, deg + 1))
            for k in range(nk):
                for i in range(ni):
                    for j in range(deg + 1):
                        h = t_steps[k] / ni
                        T[k, i, j] = experiment.fe_time[k] + h * (i + tau_root[j])
        
            lagrange_polynomials = []
            for j in range(deg + 1):
                L = 1
                for r in range(deg + 1):
                    if r != j:
                        L *= (tau - tau_root[r]) / (tau_root[j] - tau_root[r])
                lfcn = castools.Function("lfcn", [tau], [L])
                lagrange_polynomials.append(lfcn)
                D[j] = lfcn(1.0)
                tfcn = castools.Function("tfcn", [tau], [castools.tangent(L, tau)])
                for r in range(deg + 1):
                    C[j, r] = tfcn(tau_root[r])
        
            # Define symbolic variables for collocation
            xk0 = self.model.sv.sym("xk0", n_x)
            pk = self.model.sv.sym("pk", n_p)
            tv_pk = self.model.sv.sym("tv_pk", n_tvp)
            uk = self.model.sv.sym("uk", n_u)
            wk = self.model.sv.sym("wk", n_w)
        
            # State trajectory
            n_ik = ni * (deg + 1) * n_x
            ik = self.model.sv.sym("ik", n_ik)
        
            ik_split = np.resize(np.array([], dtype=self.model.sv.dtype), (ni, deg + 1))
            offset = 0
        
            # Algebraic trajectory
            n_zk = ni * (deg + 1) * n_z
            zk = self.model.sv.sym("zk", n_zk)
            offset_z = 0
            zk_split = np.resize(np.array([], dtype=self.model.sv.dtype), (ni, deg + 1))
        
            # Store initial condition
            ik_split[0, 0] = xk0
            zk_split[0, 0] = zk[offset_z : offset_z + n_z]
            offset_z += n_z
            first_j = 1
        
            for i in range(ni):
                for j in range(first_j, deg + 1):
                    ik_split[i, j] = ik[offset : offset + n_x]
                    zk_split[i, j] = zk[offset_z : offset_z + n_z]
                    offset_z += n_z
                    offset += n_x
                first_j = 0
        
            xkf = ik[offset : offset + n_x]
            offset += n_x
            assert offset == n_ik
            assert offset_z == n_zk
        
            tau_interp = self.model.sv.sym("tau_interp")
            collocation_functions = []
        
            ifcn_k = []
            for k in range(nk):
                gk = []
                lbgk = []
                ubgk = []
                fe_interp_functions = []
        
                for i in range(ni):
                    a_i0 = afcn(ik_split[i, 0], uk, zk_split[i, 0], tv_pk, pk, wk)
                    gk.append(a_i0)
                    lbgk.append(np.zeros(n_z))
                    ubgk.append(np.zeros(n_z))
        
                    for j in range(1, deg + 1):
                        xp_ij = 0
                        for r in range(deg + 1):
                            xp_ij += C[r, j] * ik_split[i, r]
        
                        f_ij = ffcn(ik_split[i, j], uk, zk_split[i, j], tv_pk, pk, wk)
                        h = t_steps[k] / ni
                        gk.append(h * f_ij - xp_ij)
                        lbgk.append(np.zeros(n_x))
                        ubgk.append(np.zeros(n_x))
        
                        a_ij = afcn(ik_split[i, j], uk, zk_split[i, j], tv_pk, pk, wk)
                        gk.append(a_ij)
                        lbgk.append(np.zeros(n_z))
                        ubgk.append(np.zeros(n_z))
        
                    xf_i = 0
                    for r in range(deg + 1):
                        xf_i += D[r] * ik_split[i, r]
        
                    x_next = ik_split[i + 1, 0] if i + 1 < ni else xkf
                    gk.append(x_next - xf_i)
                    lbgk.append(np.zeros(n_x))
                    ubgk.append(np.zeros(n_x))
        
                    x_i_tau = np.resize(np.array([], dtype=self.model.sv.dtype), (n_x,))
        
                    for j in range(deg + 1):
                        if i == 0 and j == 0:
                            x_ij = xk0
                        else:
                            x_ij = ik_split[i, j]
                        x_i_tau += lagrange_polynomials[j](tau_interp) * x_ij
        
                    fe_interp_fun = castools.Function(
                        f"interp_fun_k{k}_fe{i}", [tau_interp, xk0, ik], [x_i_tau]
                    )
                    fe_interp_functions.append(fe_interp_fun)
        
                collocation_functions.append(fe_interp_functions)
        
                gk = castools.vertcat(*gk)
                lbgk = np.concatenate(lbgk)
                ubgk = np.concatenate(ubgk)
        
                assert gk.shape[0] == ik.shape[0] + zk.shape[0]
        
                ifcn = castools.Function("ifcn", [xk0, ik, uk, zk, tv_pk, pk, wk], [gk, xkf])
                ifcn_k.append(ifcn)
        
        return ifcn_k, n_total_coll_points, collocation_functions

    def _update_bounds(self):
        """Update optimization variable bounds from estimator bound structs."""
        
        for j, experiment in enumerate(self.experiment_list):
            if self.settings.cons_check_colloc_points:
                self._lb_opt_x[f"_x_exp_{j}"] = self._physical_to_optimizer(
                    self._x_lb.cat, self._x_scaling, self._x_offset
                )
                self._ub_opt_x[f"_x_exp_{j}"] = self._physical_to_optimizer(
                    self._x_ub.cat, self._x_scaling, self._x_offset
                )
                self._lb_opt_x[f"_z_exp_{j}"] = self._physical_to_optimizer(
                    self._z_lb.cat, self._z_scaling, self._z_offset
                )
                self._ub_opt_x[f"_z_exp_{j}"] = self._physical_to_optimizer(
                    self._z_ub.cat, self._z_scaling, self._z_offset
                )
            else:
                raise NotImplementedError(
                    "Constraints are only available for all collocation points."
                )
        
            self._lb_opt_x[f"_u_exp_{j}"] = self._physical_to_optimizer(
                self._u_lb.cat, self._u_scaling, self._u_offset
            )
            self._ub_opt_x[f"_u_exp_{j}"] = self._physical_to_optimizer(
                self._u_ub.cat, self._u_scaling, self._u_offset
            )
        
            # Pin `_u` pass-through channels to FE-aligned data. The
            # measurement-equality rows for these channels were dropped in
            # _prepare_nlp; bounds replace them as the mechanism that "replays
            # the input verbatim from the CSV." Pinning is per-FE per-name in
            # optimizer space (apply scaling/offset elementwise via
            # _physical_to_optimizer over a single-element vector).
            for y_idx, u_name in self._u_passthrough_meas:
                u_data = experiment.u_interp.get(u_name)
                if u_data is None:
                    continue
                u_scale = float(self._u_scaling[u_name])
                u_off = float(self._u_offset[u_name])
                for k in range(experiment.n_horizon):
                    val_phys = float(u_data[k])
                    val_opt = (val_phys - u_off) / u_scale
                    self._lb_opt_x[f"_u_exp_{j}", k, u_name] = val_opt
                    self._ub_opt_x[f"_u_exp_{j}", k, u_name] = val_opt
        
            self._lb_opt_x[f"_eps_exp_{j}"] = self._eps_lb.cat
            self._ub_opt_x[f"_eps_exp_{j}"] = self._eps_ub.cat
        
        self._lb_opt_x["_p_est"] = self._p_est_to_optimizer(
            self._p_est_lb.cat, label="Lower bound"
        )
        self._ub_opt_x["_p_est"] = self._p_est_to_optimizer(
            self._p_est_ub.cat, label="Upper bound"
        )

    def solve(self):
        """Solve the NLP with IPOPT."""
        assert self.flags["setup"], "Backend was not setup yet."
        
        solver_call_kwargs = {
            "x0": self._opt_x_num,
            "lbx": self._lb_opt_x,
            "ubx": self._ub_opt_x,
            "lbg": self._nlp_cons_lb,
            "ubg": self._nlp_cons_ub,
            "p": self._opt_p_num,
        }
        
        if self.flags["initial_run"]:
            solver_call_kwargs.update(
                {
                    "lam_x0": self.lam_x_num,
                    "lam_g0": self.lam_g_num,
                }
            )
        
        r = self.S(**solver_call_kwargs)
        self._opt_x_num.master = r["x"]
        unscaled = self._optimizer_to_physical(
            r["x"], self.opt_x_scaling.master, self.opt_x_offset.master
        )
        if self._p_est_log_mask.any():
            p_lo, p_hi = self._p_est_opt_x_range
            unscaled[p_lo:p_hi] = self._p_est_from_optimizer(r["x"][p_lo:p_hi])
        self.opt_x_num_unscaled.master = unscaled
        self.opt_g_num = r["g"]
        self.lam_g_num = r["lam_g"]
        self.lam_x_num = r["lam_x"]
        self.solver_stats = self.S.stats()
        
        self.opt_aux_num.master = self.opt_aux_expression_fun(self._opt_x_num, self._opt_p_num)
        
        self.flags["initial_run"] = True

    # ------------------------------------------------------------------
    #  Single Shooting
    # ------------------------------------------------------------------
    def _preprocess_experiment(self, experiment):
        """Extract numeric arrays and the measurement-to-boundary map.
        
        The integration grid is the union of ``experiment.fe_time`` and
        ``experiment.meas_time`` (see :func:`build_refined_grid`), so every
        measurement time is an integration boundary and the residual compares
        each sample against the state at exactly that time. Inputs and TVPs
        are looked up zero-order-hold from the finite element that owns each
        refined interval. The t=0 sample is kept: it is a constant w.r.t. the
        parameters when ``x0`` is fixed (harmless), but it is the most direct
        observation of ``x0`` when ``estimate_x0`` makes the initial state a
        free variable.
        """
        model = self.model
        
        x0_raw = experiment.initial_state
        x0 = np.array(x0_raw.cat if hasattr(x0_raw, "cat") else x0_raw).flatten()
        
        if model.n_z > 0:
            z0_raw = experiment.initial_algebraic
            if z0_raw is None:
                z0 = np.zeros(model.n_z)
            else:
                z0 = np.array(z0_raw.cat if hasattr(z0_raw, "cat") else z0_raw).flatten()
        else:
            z0 = np.zeros(0)
        
        fe_time = np.asarray(experiment.fe_time, dtype=float)
        n_orig_fe = len(fe_time) - 1
        
        meas_time_target = experiment.meas_time          # includes t=0
        y_meas_target = experiment.y_meas                # (n_meas, n_all_meas)
        y_mask_target = experiment.y_mask                # (n_meas, n_all_meas)
        
        t_grid, owner_interval, boundary_idx = build_refined_grid(
            fe_time, meas_time_target
        )
        n_fe = len(t_grid) - 1
        dt = np.diff(t_grid)
        
        u_seq = experiment.u[:n_orig_fe][owner_interval]      # (n_fe, n_u)
        tvp_seq = experiment.tvp[:n_orig_fe][owner_interval]  # (n_fe, n_tvp)
        
        # Interval owning each measurement boundary (for the zero-order-hold
        # u/tvp lookup in the measurement function).
        interval_idx = np.minimum(boundary_idx, max(n_fe - 1, 0))
        
        true_meas_indices = [
            list(experiment.all_meas_names).index(name) for name in experiment.y_names
        ]
        
        return {
            "n_fe": n_fe,
            "dt": dt,
            "u_seq": u_seq,
            "tvp_seq": tvp_seq,
            "x0": x0,
            "z0": z0,
            "meas_time_target": meas_time_target,
            "y_meas_target": y_meas_target,
            "y_mask_target": y_mask_target,
            "boundary_idx": boundary_idx,
            "interval_idx": interval_idx,
            "true_meas_indices": true_meas_indices,
        }

    def _get_integrator(self, dae):
        """Return the time-rescaled integrator over [0, 1], compiled once.
       
        The DAE from :func:`build_dae` multiplies the ODE by a step-length
        parameter, so one compiled instance serves every interval; each call
        passes the physical interval length as the last entry of ``p``.
        """
        if self._plugin not in self._integrator_cache:
            self._integrator_cache[self._plugin] = ca.integrator(
                "ss_step", self._plugin, dae, 0.0, 1.0, self._integrator_opts
            )
        return self._integrator_cache[self._plugin]

    def _prepare_single_shooting_nlp(self):
        """Assemble the symbolic single-shooting NLP and create the IPOPT solver."""

        model = self.model
        
        # --- Parameters: optimizer-space decision variable, physical inside
        # the graph (affine or log per parameter, via the estimator's shared
        # transform map) ---
        self._n_p_est = self.n_p_est
        
        P = ca.MX.sym("p_est", self._n_p_est)
        p_est_phys = self._p_est_from_optimizer(P)
        
        # Fixed parameters are physical (default unit scaling, matching the
        # single-shooting simulation path); combine into the full model vector.
        p_set0 = self.p_fun(0)
        p_full = self._p_cat_fun(p_est_phys, p_set0)
        
        dae = build_dae(model)
        # Weights are constant for the diagonal P_v / sigma_y path, so freezing
        # them here (at the current p_est0) is exact. A future p-dependent
        # measurement weighting would need to be threaded symbolically instead.
        weights = self._get_channel_weights()
        
        decision_vars = [P]
        lbx = [self._p_est_to_optimizer(self._p_est_lb.cat, label="Lower bound")]
        ubx = [self._p_est_to_optimizer(self._p_est_ub.cat, label="Upper bound")]
        
        estimate_x0 = self.settings.estimate_x0
        if estimate_x0:
            x_scaling = np.array(self._x_scaling.cat).flatten()
            x_offset = np.array(self._x_offset.cat).flatten()
            x_lb = np.array(self._x_lb.cat).flatten()
            x_ub = np.array(self._x_ub.cat).flatten()
        
        obj = ca.MX(0)
        for ei, ed in enumerate(self._experiment_data):
            if estimate_x0:
                X0 = ca.MX.sym(f"x0_exp_{ei}", model.n_x)
                decision_vars.append(X0)
                lbx.append((x_lb - x_offset) / x_scaling)
                ubx.append((x_ub - x_offset) / x_scaling)
                x_current = X0 * x_scaling + x_offset
            else:
                x_current = ca.DM(ed["x0"])
        
            z_current = ca.DM(ed["z0"]) if model.n_z > 0 else None
        
            x_traj = [x_current]
            z_traj = [z_current]
            for k in range(ed["n_fe"]):
                F = self._get_integrator(dae)
                tvp_k = ed["tvp_seq"][k] if model.n_tvp > 0 else np.zeros(0)
                p_intg = ca.vertcat(p_full, ca.DM(tvp_k), float(ed["dt"][k]))
                call = {
                    "x0": x_current,
                    "p": p_intg,
                    "u": ca.DM(ed["u_seq"][k].reshape(-1, 1)),
                }
                if model.n_z > 0:
                    call["z0"] = z_current
                res = F(**call)
                x_current = res["xf"]
                x_traj.append(x_current)
                if model.n_z > 0:
                    z_current = res["zf"]
                    z_traj.append(z_current)
        
            obj = obj + self._experiment_residual(ed, x_traj, z_traj, p_full, weights)

        self._nlp_x = ca.vertcat(*decision_vars)
        

        self._nlp_obj = obj
    
        self._lbx = np.concatenate([np.atleast_1d(np.array(v, dtype=float).flatten()) for v in lbx])
        self._ubx = np.concatenate([np.atleast_1d(np.array(v, dtype=float).flatten()) for v in ubx])

    def _experiment_residual(self, ed, x_traj, z_traj, p_full, weights):
        """Weighted sum of squared measurement residuals for one experiment."""
        model = self.model
        v_zero = ca.DM.zeros(model.n_v)
        tidx = ed["true_meas_indices"]
        
        total = ca.MX(0)
        for m, _ in enumerate(ed["meas_time_target"]):
            bi = int(ed["boundary_idx"][m])
            ivl = int(ed["interval_idx"][m])
        
            x_b = x_traj[bi]
            if model.n_z > 0:
                z_b = z_traj[bi] if bi < len(z_traj) else z_traj[-1]
            else:
                z_b = ca.DM.zeros(0)
        
            u_b = ca.DM(ed["u_seq"][ivl].reshape(-1, 1))
            tvp_b = ca.DM(ed["tvp_seq"][ivl]) if model.n_tvp > 0 else ca.DM.zeros(0)
        
            y_pred = model._meas_fun(x_b, u_b, z_b, tvp_b, p_full, v_zero)
        
            for j, ch in enumerate(tidx):
                mask = float(ed["y_mask_target"][m, ch])
                if mask == 0.0:
                    continue
                resid = y_pred[ch] - ed["y_meas_target"][m, ch]
                total = total + weights[j] * mask * resid ** 2
        
        return total
        
    def _get_channel_weights(self):
        """Per-true-channel least-squares weights, in ``y_names`` order.
        
        Reuses the collocation objective's measurement weighting: probes
        ``stage_cost_measurement_fun`` once per ``_v`` slot to recover
        ``diag(P_v)`` (i.e. ``1/sigma**2`` for channels declared with
        ``meas_noise=True``). Each ``_v`` slot is matched to its measurement
        channel by name (slot ``<meas_name>_noise`` weights channel
        ``<meas_name>``); channels without a noise variable receive unit
        weight. Off-diagonal ``P_v`` entries cannot be represented in the
        per-channel residual sum — a warning is emitted and the diagonal is
        used.
        """
        y_names = list(self.experiment_list[0].y_names)
        weights = np.ones(len(y_names))
        
        if not hasattr(self, "stage_cost_measurement_fun"):
            return weights
        if self.model.n_v == 0:
            return weights
        
        p_v = getattr(self, "_P_v_default", None)
        if p_v is not None and np.any(p_v - np.diag(np.diag(p_v))):
            warnings.warn(
                "P_v has off-diagonal entries; the single-shooting objective "
                "weights each measurement channel independently and uses "
                "diag(P_v) only.",
                UserWarning,
            )
        
        p_full_struct = self._p_cat_fun(self._p_est0, self.p_fun(0))
        v_names = [n for n in self.model.v.keys() if n != "default"]
        for v_name in v_names:
            base = v_name[: -len("_noise")] if v_name.endswith("_noise") else v_name
            if base not in y_names:
                continue
            v_test = self.model._v(0)
            v_test[v_name] = 1.0
            weights[y_names.index(base)] = float(
                self.stage_cost_measurement_fun(v_test, p_full_struct)
            )
        return weights

    def _solve_single_shooting(self) -> np.ndarray:
        """Solve the single-shooting NLP and return physical parameter values.

        Returns the estimated parameters as a numpy array ordered by
        ``_p_est.keys()`` (excluding the leading zero-size *default* entry), as
        expected by ``ParameterEstimator._store_estimated_params``. When
        ``estimate_x0`` is on, the recovered per-experiment initial states are
        stored in :attr:`estimated_initial_states` (physical base-SI values, one
        flat array per experiment) as a side effect.
        """
        assert self.flags.get("single_shooting_setup", False), "Single-shooting backend was not set up. Call setup() first."

        x0_guess = [
            np.asarray(
                self._p_est_to_optimizer(
                    self._p_est0.cat, label="Initial guess (p_est0)"
                )
            ).flatten()
        ]
        if self.settings.estimate_x0:
            x_scaling = np.array(self._x_scaling.cat).flatten()
            x_offset = np.array(self._x_offset.cat).flatten()
            for ed in self._experiment_data:
                x0_guess.append((ed["x0"] - x_offset) / x_scaling)
        x0_init = np.concatenate(x0_guess)

        r = self.S(x0=x0_init, lbx=self._lbx, ubx=self._ubx)
        self.solver_stats = self.S.stats()

        sol = np.array(r["x"]).flatten()
        p_est_phys = np.asarray(
            self._p_est_from_optimizer(sol[: self._n_p_est])
        ).flatten()

        if self.settings.estimate_x0:
            x_scaling = np.array(self._x_scaling.cat).flatten()
            x_offset = np.array(self._x_offset.cat).flatten()
            n_x = self.model.n_x
            self.estimated_initial_states = []
            off = self._n_p_est
            for _ in self._experiment_data:
                x0_scaled = sol[off:off + n_x]
                self.estimated_initial_states.append(x0_scaled * x_scaling + x_offset)
                off += n_x
        else:
            self.estimated_initial_states = None

        return p_est_phys

    