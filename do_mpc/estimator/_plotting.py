"""Plotting utilities for the parameter estimator.

Standalone plotting functions kept separate from ``ParameterEstimator`` so the
core estimation logic stays focused.
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi.tools as castools
from typing import List, Union, Optional


def plot_results(
    estimator,
    meas_names: List[str] = None,
    experiment_index: Union[int, List[int]] = None,
    show_fe_times: bool = True,
):
    """Plot the results of the parameter estimation.

    The results are plotted for each experiment in the list of experiments.
    The results are plotted for each measurement variable.

    Args:
        estimator: The ParameterEstimator instance with completed estimation.
        meas_names: List of measurement names to plot. If None, all measurements are plotted.
        experiment_index: Index or list of indices of experiments to plot. If None, all experiments are plotted.
        show_fe_times: If True, indications for finite element time points are added to the plots.

    Returns:
        List of (fig, ax) tuples, one per plotted experiment.
    """
    if estimator.flags["initial_run"] == False:
        raise Exception(
            "The parameter estimation was not run yet. Please call .estimate_parameters() first."
        )

    if meas_names is None:
        meas_names = estimator.model._y.keys()[1:]

    if experiment_index is None:
        experiment_index = list(range(len(estimator.experiment_list)))
    elif isinstance(experiment_index, int):
        experiment_index = [experiment_index]

    fig_axs = []
    for j, (
        experiment,
        (ifcn_k, n_total_coll_points, collocation_functions),
    ) in enumerate(zip(estimator.experiment_list, estimator.discretization_list)):
        if j not in experiment_index:
            continue

        y_calc_list = []
        y_calc_noise_free_list = []
        # calculate the measurement values based on the optimization results
        for m in range(experiment.n_meas):
            fe_idx = experiment.meas_fe_indices[m]
            sub_element_idx = experiment.meas_sub_element_indices[m]
            tau_value = experiment.meas_tau_values[m]

            # Get the collocation function for this measurement
            collocation_function = collocation_functions[fe_idx][sub_element_idx]

            x_interp = collocation_function(
                tau_value,
                estimator.opt_x_num[
                    f"_x_exp_{j}", fe_idx, -1
                ],  # Initial state for this element
                castools.vertcat(*estimator.opt_x_num[f"_x_exp_{j}", fe_idx + 1, :-1]),
            )  # Collocation points
            x_interp = estimator._optimizer_to_physical(
                x_interp, estimator._x_scaling.cat, estimator._x_offset.cat
            )

            p = estimator._p_cat_fun(
                estimator._p_est_from_optimizer(estimator.opt_x_num[f"_p_est"]),
                estimator._optimizer_to_physical(
                    estimator.opt_p_num[f"_p_set"],
                    estimator._p_set_scaling,
                    estimator._p_set_offset,
                ),
            )

            y_calc = estimator.model._meas_fun(
                x_interp,
                estimator._optimizer_to_physical(
                    estimator.opt_x_num[f"_u_exp_{j}", fe_idx],
                    estimator._u_scaling.cat,
                    estimator._u_offset.cat,
                ),
                estimator._optimizer_to_physical(
                    estimator.opt_x_num[f"_z_exp_{j}", fe_idx, 0],
                    estimator._z_scaling.cat,
                    estimator._z_offset.cat,
                ),
                estimator.opt_p_num[f"_tvp_exp_{j}", fe_idx],
                p,
                estimator.opt_x_num[f"_v_exp_{j}", m],
            )

            y_calc_noise_free = estimator.model._meas_fun(
                x_interp,
                estimator._optimizer_to_physical(
                    estimator.opt_x_num[f"_u_exp_{j}", fe_idx],
                    estimator._u_scaling.cat,
                    estimator._u_offset.cat,
                ),
                estimator._optimizer_to_physical(
                    estimator.opt_x_num[f"_z_exp_{j}", fe_idx, 0],
                    estimator._z_scaling.cat,
                    estimator._z_offset.cat,
                ),
                estimator.opt_p_num[f"_tvp_exp_{j}", fe_idx],
                p,
                np.zeros(estimator.opt_x_num[f"_v_exp_{j}", m].shape),
            )

            y_calc_list.append(y_calc)
            y_calc_noise_free_list.append(y_calc_noise_free)

        y_calc = np.array(y_calc_list)
        y_calc_noise_free = np.array(y_calc_noise_free_list)
        y_meas = np.array(estimator.opt_p_num[f"_y_meas_exp_{j}"])
        y_mask = np.array(estimator.opt_p_num[f"_y_mask_exp_{j}"])

        y_meas[y_mask == 0] = np.nan

        fig, ax = plt.subplots(
            len(meas_names), 1, figsize=(10, 5 * len(meas_names)), sharex=True
        )
        if len(meas_names) == 1:
            ax = [ax]
        for i, meas_name in enumerate(meas_names):
            # TODO: the following only works if all measurements are considered as noisy.
            ax[i].plot(
                experiment.meas_time,
                y_calc_noise_free[:, i],
                label="Calculated (noise free)",
                color="green",
            )
            ax[i].scatter(
                experiment.meas_time,
                y_meas[:, i],
                label="Measured",
                color="red",
                marker="x",
            )
            ax[i].plot(
                experiment.meas_time,
                y_calc[:, i],
                label="Calculated",
                color="blue",
                linestyle="--",
            )
            ax[i].set_title(f"Experiment {j + 1}: {meas_name}")
            ax[i].set_ylabel(meas_name)
            ax[i].legend()

            # Add vertical lines for finite element boundaries if requested
            if show_fe_times:
                for fe_time in experiment.fe_time:
                    ax[i].axvline(
                        x=fe_time,
                        color="lightgray",
                        linestyle="-",
                        alpha=0.5,
                        linewidth=0.5,
                    )

        # Add ticks at finite element times if requested
        if show_fe_times:
            if len(experiment.fe_time) <= 20:
                plt.xticks(experiment.fe_time)
            else:
                step = len(experiment.fe_time) // 20 + 1
                plt.xticks(experiment.fe_time[::step])

            ax[-1].set_xlabel(
                "Time (vertical lines indicate finite element boundaries)"
            )
        else:
            ax[-1].set_xlabel("Time")

        plt.tight_layout()
        fig_axs.append((fig, ax))

    return fig_axs
