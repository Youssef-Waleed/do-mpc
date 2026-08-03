single shooting: choose parameter value and use integrator to generate the whole trajectory and then use a cost function to compare the estimated trajectory with the actual trajectory

multiple shooting: dive the trajectory into small chunks and add a continuity constraint to ensure each chunk has as initial condition the end point of the last chunk.  

multiple-experiments: Minimize the trajectory estimation error over multiple trajectories. So choose the parameter set that minimizes the error across multiple trajectories (different starting points, different inputs, etc...)



override:

bounds setting and updating

prepare_nlp

integrator (for single shooting)


explain:

_prepare_nlp():

Lines 105-123:
loops through every dataset and creates completely isolated blocks of variables in memory: `_x_exp_0`, `_x_exp_1`, etc.

-Orthogonal Collocation setup

-The Continuity constraint

-Interpolation of asynchronous data measurements using Lagrange polynomials:

The collocation points/grid points are used to produce an expression in terms of the optimization variables for the real state at a certain measurement's timestamp.
That state is then fed into the measurement model h(x,u,p) in order to obtain an estimate of what the sensor would read given this state. This is then compared with actual measurement to calculate the residual. 

(also creates the parallel branches structure in the optimization problem. this part would be hard to replace with a call of the tree function in the optimizer class since the branches we need to create don't share the same root.

_setup_discretization():

is just the same method in optimizer.py except it also returns the lagrange polynomials not just the equality constraints

suggested: add a flag to optimizer's _setup_discretization() so it can also return the polynomials if needed 

eg: def _setup_discretization(self, return_interpolators=False):


The set_initial_guess() method uses an integrator and runs a fake forward simulation for every single experiment using the initial parameter guess (p_est0). It then takes the output of that simulation and uses it to warm start the casadi optimzation.
 

Update_bounds:

uses the exact same technique as MHE (bound pinning) to assign known input values to the input optimization variable to force casadi to ignore them.

we could use TVPs to embed the input array into the dynamic model instead of bound pinning but this would introduce the need for a separate model instance for control and one for estimation. 


===========================================================================================================================================================================================


Batch_optimizer parent class implementation:


The Old Architecture: The `ParameterEstimator` forwarded commands to two worker classes: `CollocationBackend` and `SingleShootingBackend`. 

The New Architecture: The mathematical execution has been unified into a base class called `BatchOptimizer`. The `ParameterEstimator` now directly inherits from this class.


Structural Mapping: Where Did the Functions Go?

The refactor cleanly divides responsibilities. `BatchOptimizer` is now the mathematical engine, while `ParameterEstimator` is the user-facing configuration and analysis tool.

`BatchOptimizer`:

* *Collocation Logic*: `_prepare_nlp()`, `_create_nlp()`, and `_setup_discretization()`


* *Single-Shooting Logic*: `_preprocess_experiment()`, `_prepare_single_shooting_nlp()`, `_experiment_residual()`, and `_get_integrator()`


* *Solvers*: The execution commands for both methods—`solve()` (for Collocation) and `_solve_single_shooting()`


* *NLP Structures:* All CasADi structural initialization (`_opt_x`, `_opt_p_num`, `_nlp_obj`, `_nlp_cons`) is managed by `BatchOptimizer`.


* *Experiment Storage:* The `add_experiment()` function was moved to the base class to manage the `experiment_list` centrally.



`ParameterEstimator`:

* *Setup:*  (`set_nl_cons`), bounds (`bounds`), scaling (`scaling`), and transforms (`transform`).


* *Objective:* creation of objective function (`set_objective`, `set_default_objective`).


* *The Dispatcher:* `estimate_parameters()` inherits solver methods from `BatchOptimizer`.


* *Post-Processing & Validation:*  `simulate_experiments()`, `evaluate_fit()`, `plot_results()`, and identifiability is in `ParameterEstimator`.



This removed `ParameterEstimator`'s need for a backend class (e.g., `def opt_x_num(self): return self._backend.opt_x_num`).




