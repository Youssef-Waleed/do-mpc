import numpy as np
import pandas as pd
import do_mpc
import warnings
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass
from do_mpc.estimator._utils import (
    find_closest_time_index,
    validate_data_columns,
    consolidate_nearby_timepoints,
    interpolate_signal_at_index,
)


@dataclass
class ExperimentSettings:
    """Settings for experiment configuration."""

    collocation_type: str | None = None
    collocation_deg: int | None = None
    collocation_ni: int | None = 1
    t_step: float | None = None


class Experiment:
    """Class for setting up and managing experiments with do_mpc models.

    This class handles data processing, interpolation, and discretization
    for parameter estimation experiments.
    """

    def __init__(self, model: do_mpc.model.Model, experiment_type: str = "dynamic"):
        """Initialize the experiment with a do_mpc model."""
        self.model = model
        self.settings = ExperimentSettings()
        self.flags = {"setup": False, "settings_specified": False}

        # Extract model variable names
        self.all_meas_names = model.y.keys()[1:]  # All measurement names including inputs
        self.u_names = model.u.keys()[1:]
        self.tvp_names = model.tvp.keys()[1:] if hasattr(model, "tvp") else []

        # Validate that all input names are included in measurement names
        missing_inputs_in_measurements = [
            u_name for u_name in self.u_names if u_name not in self.all_meas_names
        ]
        if missing_inputs_in_measurements:
            raise AssertionError(
                f"Input variables {missing_inputs_in_measurements} must be included in measurement variables."
            )

        # Create a separate list for true measurements (excluding inputs)
        self.y_names = [y_name for y_name in self.all_meas_names if y_name not in self.u_names]

        self.experiment_type = experiment_type

        # Will be set during setup
        self.data = None
        self.fe_time = None  # Finite element time points
        self.n_horizon = None
        self.initial_state = None
        self.initial_algebraic = None

        # For storing interpolated values
        self.u_interp = None
        self.tvp_interp = None

        self.x_guess = None
        self.u_guess = None
        self.z_guess = None
        self.ignored_y_names: List[str] = []

    def set_settings(self, settings: Dict[str, Any]) -> None:
        """Set the experiment settings.

        Parameters
        ----------
        settings : Dict[str, Any]
            Dictionary containing experiment settings

        Raises
        ------
        AssertionError
            If required settings are missing or invalid
        """
        valid_keys = set(ExperimentSettings.__dataclass_fields__.keys())
        unknown = sorted(set(settings.keys()) - valid_keys)
        assert not unknown, (
            f"Unknown experiment setting(s): {unknown}. "
            f"Known keys: {sorted(valid_keys)}"
        )

        # Validate settings
        if self.experiment_type == "dynamic":
            assert "collocation_type" in settings, "Collocation type is required."
            assert "collocation_deg" in settings, "Collocation degree is required."
            assert "t_step" in settings, "Time step is required."

        # Reset to schema defaults, then apply user overrides.
        self.settings = ExperimentSettings()

        # Set the settings
        for key, value in settings.items():
            setattr(self.settings, key, value)

        # Stash raw settings dict so downstream tooling (e.g. test-experiment
        # rebuild in rl_training.test_simulation) can recover them without
        # depending on internal attribute names.
        self._settings = {
            key: getattr(self.settings, key)
            for key in valid_keys
            if getattr(self.settings, key) is not None
        }

        # Mark settings as specified
        self.flags["settings_specified"] = True
        # Reset setup flag if it was already set
        self.flags["setup"] = False

    def validate_mandatory_settings(self) -> None:
        """Validate that all mandatory settings are specified."""
        if not self.flags["settings_specified"]:
            raise AssertionError("Experiment settings must be specified before setup.")

    def setup(
        self,
        data: pd.DataFrame,
        initial_state: Optional[np.ndarray] = None,
        change_threshold: Union[float, Dict[str, float]] = 0.01,
        initial_algebraic: Optional[np.ndarray] = None,
    ) -> None:
        """Setup the experiment with the given data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the experiment data
        initial_state : Optional[np.ndarray]
            Initial state vector for dynamic experiments
        change_threshold : Union[float, Dict[str, float]], optional
            Threshold for considering an input/tvp change significant.
            Can be a single float for all signals or a dictionary mapping
            signal names to individual thresholds, by default 0.01
        initial_algebraic : Optional[np.ndarray], optional
            Initial algebraic state vector, by default None

        Raises
        ------
        AssertionError
            If required columns are missing or initial state is not provided
        """
        # Validate settings
        self.validate_mandatory_settings()

        if not data.columns.is_unique:
            dupes = sorted(set(data.columns[data.columns.duplicated()].tolist()))
            raise ValueError(
                f"Experiment data has duplicate column names: {dupes}. "
                f"Each column name must be unique before calling Experiment.setup."
            )

        # Store the data and make a copy to avoid modifying the original
        self.data = data.copy()

        # Validate data columns and auto-mask missing true-measurement channels
        validation_result = self._validate_data_columns()
        self._apply_missing_measurement_workaround(validation_result)

        if self.experiment_type == "dynamic":
            self._setup_dynamic_experiment(initial_state, change_threshold, initial_algebraic)

        # Mark experiment as set up
        self.flags["setup"] = True

    def _validate_data_columns(self) -> dict:
        """Validate that all required columns are present in the data."""
        return validate_data_columns(
            self.data, self.y_names, self.u_names, self.tvp_names, self.experiment_type
        )

    def _apply_missing_measurement_workaround(self, validation_result: dict) -> None:
        """Exclude measurement channels that have no corresponding CSV column.

        Channels listed in ``ignored_y_names`` are excluded from the estimation
        objective by leaving their mask entries at zero.  No dummy columns are
        injected into ``self.data``.
        """
        missing_outputs = validation_result.get("missing_outputs", [])
        self.ignored_y_names = list(missing_outputs)
        if not missing_outputs:
            return

        for col in missing_outputs:
            warnings.warn(
                f"Measurement channel '{col}' is declared via set_meas but has no "
                f"corresponding column in the data. It has been excluded from the "
                f"estimation objective. To include it, add a '{col}' column to your "
                f"CSV; to suppress this warning, remove the set_meas('{col}', ...) "
                f"call from the model.",
                UserWarning,
                stacklevel=3,
            )

    def _normalize_time_column(self) -> None:
        """Normalize the time column to start from zero."""
        self.data["time"] = self.data["time"] - self.data["time"].iloc[0]

    def _detect_significant_changes(
        self, change_threshold: Union[float, Dict[str, float]] = 0.01
    ) -> np.ndarray:
        """Detect timepoints where inputs or TVPs change significantly.

        Parameters
        ----------
        change_threshold : Union[float, Dict[str, float]], optional
            Threshold for considering a change significant

        Returns
        -------
        numpy.ndarray
            Array of time points where significant changes occur
        """
        # Get monitored signals (inputs and TVPs)
        monitored_signals = self.u_names + self.tvp_names

        # Convert threshold to dictionary if it's a single value
        if isinstance(change_threshold, (int, float)):
            thresholds = {signal: change_threshold for signal in monitored_signals}
        else:
            # Use provided thresholds, with default for signals not specified
            thresholds = {
                signal: change_threshold.get(signal, 0.01) for signal in monitored_signals
            }

        # Store the time points where significant changes are detected
        change_times = [0.0]  # Always include the first time point

        for signal in monitored_signals:
            signal_data = self.data[signal]

            # Skip signals with all NaN values
            if signal_data.isna().all():
                continue

            # Calculate the signal range for normalization
            valid_data = signal_data.dropna()
            if len(valid_data) < 2:
                continue

            signal_range = valid_data.max() - valid_data.min()

            # Avoid division by zero for constant signals
            if signal_range < 1e-10:
                continue

            # Calculate normalized absolute differences
            # Handle NaNs by forward-filling where possible
            filled_data = signal_data.ffill()  # Updated from fillna(method='ffill')
            # There might still be NaNs at the beginning
            filled_data = filled_data.bfill()  # Updated from fillna(method='bfill')

            diffs = np.abs(filled_data.diff()) / signal_range

            # Find where changes exceed the threshold
            significant_changes = diffs > thresholds[signal]
            change_idx = self.data.index[significant_changes].tolist()

            # Add the corresponding time points
            if change_idx:
                change_times.extend(self.data.loc[change_idx, "time"].values)

        # Remove duplicates and sort
        change_times = np.unique(change_times)

        # Consolidate nearby time points
        change_times = self._consolidate_nearby_timepoints(change_times)

        return change_times

    def _consolidate_nearby_timepoints(
        self, timepoints: np.ndarray, min_distance: float = 1e-5
    ) -> np.ndarray:
        """Consolidate timepoints that are very close to each other."""
        return consolidate_nearby_timepoints(timepoints, min_distance)

    def _create_time_grid(self, change_threshold: Union[float, Dict[str, float]] = 0.01) -> None:
        """Create the finite element time grid.

        Parameters
        ----------
        change_threshold : Union[float, Dict[str, float]], optional
            Threshold for detecting significant changes
        """
        # Get time points where inputs or TVPs change significantly
        change_times = self._detect_significant_changes(change_threshold)

        # Find the last measurement time to use as final time
        last_measurement_time = self.data["time"].iloc[-1]
        self.final_time = last_measurement_time

        # Ensure change_times includes the final time
        if not np.isclose(change_times[-1], self.final_time):
            change_times = np.append(change_times, self.final_time)

        # Initialize the list of finite element times with the change times
        fe_times = list(change_times)

        # For each pair of consecutive change times, add equidistant points if needed
        for i in range(len(change_times) - 1):
            start_time = change_times[i]
            end_time = change_times[i + 1]
            time_span = end_time - start_time

            # If the gap between change times is larger than t_step, add intermediate points
            if time_span > self.settings.t_step:
                # Calculate number of steps needed
                n_steps = int(np.ceil(time_span / self.settings.t_step))
                # Create equidistant points, excluding start and end which are already in fe_times
                if n_steps > 1:  # Only add intermediate points if we need more than 1 step
                    intermediate_points = np.linspace(start_time, end_time, n_steps + 1)[1:-1]
                    fe_times.extend(intermediate_points)

        # Sort all times and ensure no duplicates
        self.fe_time = np.sort(np.array(fe_times))
        self.n_horizon = len(self.fe_time) - 1  # Number of intervals

    def _interpolate_signals(self) -> None:
        """Interpolate input and TVP values at the finite element time points.

        Uses zero-order hold interpolation with NaN backward/forward fill.
        """
        self.u_interp = {u_name: np.zeros(len(self.fe_time)) for u_name in self.u_names}
        self.tvp_interp = {tvp_name: np.zeros(len(self.fe_time)) for tvp_name in self.tvp_names}

        original_time = self.data["time"].values

        for i, t in enumerate(self.fe_time):
            idx = self._find_closest_time_index(original_time, t)

            for u_name in self.u_names:
                self.u_interp[u_name][i] = interpolate_signal_at_index(self.data, u_name, idx)

            for tvp_name in self.tvp_names:
                self.tvp_interp[tvp_name][i] = interpolate_signal_at_index(self.data, tvp_name, idx)

    def _find_closest_time_index(self, original_time: np.ndarray, t: float) -> int:
        """Find the index of the closest time point in the original data."""
        return find_closest_time_index(original_time, t)

    def _create_arrays(self) -> None:
        """Create arrays for use in the parameter estimator."""
        # Create arrays for inputs and TVPs
        self.u = np.zeros((len(self.fe_time), len(self.u_names)))
        self.tvp = np.zeros((len(self.fe_time), len(self.tvp_names)))

        # Fill the arrays with the interpolated values
        for i, t in enumerate(self.fe_time):
            for j, u_name in enumerate(self.u_names):
                self.u[i, j] = self.u_interp[u_name][i]

            for j, tvp_name in enumerate(self.tvp_names):
                self.tvp[i, j] = self.tvp_interp[tvp_name][i]

    def _setup_measurements(self) -> None:
        """Setup measurement arrays for the parameter estimator.

        Creates arrays for storing measurement values and binary masks
        indicating whether measurements are valid at each time point.
        """
        # Initialize arrays for measurements and masks
        self.y_meas = np.zeros((len(self.fe_time), len(self.y_names)))
        self.y_mask = np.zeros((len(self.fe_time), len(self.y_names)), dtype=int)

        # Get the original time points from the dataframe
        original_time = self.data["time"].values

        # For each finite element time point, find the closest measurement
        for i, t in enumerate(self.fe_time):
            # Find index of closest time point in original data
            idx = self._find_closest_time_index(original_time, t)

            # Store measurement values and masks
            for j, y_name in enumerate(self.y_names):
                if y_name in self.ignored_y_names:
                    # Channel excluded — no CSV column; mask stays 0
                    self.y_meas[i, j] = 0.0
                    self.y_mask[i, j] = 0
                    continue
                value = self.data[y_name].iloc[idx]

                # Check if the measurement exists (not NaN)
                if not np.isnan(value):
                    self.y_meas[i, j] = value
                    self.y_mask[i, j] = 1  # Valid measurement
                else:
                    self.y_meas[i, j] = 0.0  # Default value
                    self.y_mask[i, j] = 0  # Invalid/missing measurement

    def _find_measurement_timepoints(self) -> np.ndarray:
        """Find all timepoints where at least one measurement or input has a non-NaN value.

        Returns
        -------
        np.ndarray
            Array of timepoints with valid measurements or inputs
        """
        # Get all column names for measurements and inputs, excluding ignored y channels
        all_signal_names = [
            n for n in self.all_meas_names + self.tvp_names
            if n not in self.ignored_y_names
        ]

        # Create a mask for rows where at least one signal is not NaN
        valid_rows = self.data[all_signal_names].notna().any(axis=1)

        # Extract the corresponding timepoints
        meas_timepoints = self.data.loc[valid_rows, "time"].values

        # Sort and ensure unique values
        meas_timepoints = np.sort(np.unique(meas_timepoints))

        return meas_timepoints

    def _map_timepoints_to_elements(self, timepoints: np.ndarray) -> tuple:
        """Map timepoints to finite elements, sub-elements, and calculate tau values.

        Parameters
        ----------
        timepoints : np.ndarray
            Array of timepoints to map

        Returns
        -------
        tuple
            (fe_indices, sub_element_indices, tau_values)
        """
        n_timepoints = len(timepoints)
        fe_indices = np.zeros(n_timepoints, dtype=int)
        sub_element_indices = np.zeros(n_timepoints, dtype=int)
        tau_values = np.zeros(n_timepoints)

        # Number of sub-elements per finite element
        ni = self.settings.collocation_ni

        for i, t in enumerate(timepoints):
            # Find the index of the finite element containing this timepoint
            fe_idx = np.searchsorted(self.fe_time, t, side="right") - 1

            # Ensure index is valid (in case of numerical issues)
            fe_idx = max(0, min(fe_idx, len(self.fe_time) - 2))

            # Calculate position within the finite element (0 to 1)
            fe_start = self.fe_time[fe_idx]
            fe_end = self.fe_time[fe_idx + 1]
            fe_duration = fe_end - fe_start

            # Ensure no division by zero
            if fe_duration > 1e-10:
                position_in_fe = (t - fe_start) / fe_duration
            else:
                # For very small elements, set position based on relative proximity
                if abs(t - fe_start) < abs(fe_end - t):
                    position_in_fe = 0.0
                else:
                    position_in_fe = 1.0

            # Determine which sub-element contains this timepoint
            # Each sub-element covers 1/ni of the finite element
            sub_elem_idx = min(int(position_in_fe * ni), ni - 1)

            # Calculate tau within the sub-element
            # Each sub-element spans 1/ni of the total duration
            sub_elem_start = position_in_fe * ni - sub_elem_idx

            # Normalize tau to [0,1] within the sub-element
            tau = min(max(sub_elem_start * ni, 0.0), 1.0)

            # Store the results
            fe_indices[i] = fe_idx
            sub_element_indices[i] = sub_elem_idx
            tau_values[i] = tau

        return fe_indices, sub_element_indices, tau_values

    def _setup_measurements_for_interpolation(self) -> None:
        """Setup measurement arrays for interpolation with collocation polynomials.

        This identifies all timepoints with valid measurements or inputs,
        maps them to finite elements, and prepares arrays for measurement interpolation.
        """
        # Find all timepoints with valid measurements or inputs
        self.meas_time = self._find_measurement_timepoints()
        self.n_meas = len(self.meas_time)

        # Map these timepoints to finite elements and calculate tau values
        self.meas_fe_indices, self.meas_sub_element_indices, self.meas_tau_values = (
            self._map_timepoints_to_elements(self.meas_time)
        )

        # Initialize arrays for measurements and masks
        # Use all_meas_names which includes both true measurements and inputs
        self.y_meas = np.zeros((self.n_meas, len(self.all_meas_names)))
        self.y_mask = np.zeros((self.n_meas, len(self.all_meas_names)), dtype=int)

        # Original time points from the dataframe
        original_time = self.data["time"].values

        # For each measurement time point, find the closest point in original data
        for i, t in enumerate(self.meas_time):
            # Find exact match or closest point
            idx = np.argmin(np.abs(original_time - t))

            # Store measurement values and masks for all measurement variables
            for j, y_name in enumerate(self.all_meas_names):
                if y_name in self.ignored_y_names:
                    # Channel excluded — no CSV column; mask stays 0
                    self.y_meas[i, j] = 0.0
                    self.y_mask[i, j] = 0
                    continue
                value = self.data[y_name].iloc[idx]

                # Check if the measurement exists (not NaN)
                if not np.isnan(value):
                    self.y_meas[i, j] = value
                    self.y_mask[i, j] = 1  # Valid measurement
                else:
                    self.y_meas[i, j] = 0.0  # Default value
                    self.y_mask[i, j] = 0  # Invalid/missing measurement

    def _setup_dynamic_experiment(
        self,
        initial_state: np.ndarray,
        change_threshold: Union[float, Dict[str, float]] = 0.01,
        initial_algebraic: Optional[np.ndarray] = None,
    ) -> None:
        """Setup a dynamic experiment."""
        # Validate initial state
        if initial_state is None:
            raise AssertionError("Initial state must be provided for dynamic experiments.")

        self.initial_state = initial_state
        self.initial_algebraic = initial_algebraic

        # Normalize time column to start from 0
        self._normalize_time_column()

        # Detect significant changes and create time grid
        self._create_time_grid(change_threshold)

        # Interpolate inputs and TVPs at grid points (for initial guesses)
        self._interpolate_signals()

        # Create arrays for parameter estimator initial guesses
        self._create_arrays()

        # Setup measurement arrays for interpolation
        self._setup_measurements_for_interpolation()
