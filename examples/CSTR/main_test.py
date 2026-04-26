import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
from matplotlib.lines import Line2D
import os
import sys

rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer

# local imports
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

show_animation = True
store_results = False

print("--- INITIALIZING (Default) ---")
model_1 = template_model()
mpc_1 = template_mpc(model_1) # Passes a standard float
simulator_1 = template_simulator(model_1)
estimator_1 = do_mpc.estimator.StateFeedback(model_1)

print("--- INITIALIZING (User-Defined) ---")
model_2 = template_model()
mpc_2 = template_mpc(model_2, test_flag = True) # Passes a CasADi symbol
simulator_2 = template_simulator(model_2)
estimator_2 = do_mpc.estimator.StateFeedback(model_2)

# Set the initial state
C_a_0 = 0.8 
C_b_0 = 0.5 
T_R_0 = 134.14 
T_K_0 = 130.0 
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

# Push initial condition to all 
mpc_1.x0 = x0
simulator_1.x0 = x0
mpc_2.x0 = x0
simulator_2.x0 = x0

# Initial guesses
mpc_1.set_initial_guess()
simulator_1.set_initial_guess()
mpc_2.set_initial_guess()
simulator_2.set_initial_guess()

# --- GRAPHICS SETUP ---
graphics_1 = do_mpc.graphics.Graphics(mpc_1.data)
graphics_2 = do_mpc.graphics.Graphics(mpc_2.data)

fig, ax = plt.subplots(5, sharex=True, figsize=(10, 8))

# Add Constant lines (Solid)
graphics_1.add_line(var_type='_x', var_name='C_a', axis=ax[0], color='tab:blue')
graphics_1.add_line(var_type='_x', var_name='C_b', axis=ax[0], color='tab:orange')
graphics_1.add_line(var_type='_x', var_name='T_R', axis=ax[1], color='tab:blue')
graphics_1.add_line(var_type='_x', var_name='T_K', axis=ax[1], color='tab:orange')
graphics_1.add_line(var_type='_aux', var_name='T_dif', axis=ax[2], color='tab:blue')
graphics_1.add_line(var_type='_u', var_name='Q_dot', axis=ax[3], color='tab:blue')
graphics_1.add_line(var_type='_u', var_name='F', axis=ax[4], color='tab:blue')

# Add Symbolic lines (Dashed)
graphics_2.add_line(var_type='_x', var_name='C_a', axis=ax[0], color='black', linestyle='--')
graphics_2.add_line(var_type='_x', var_name='C_b', axis=ax[0], color='red', linestyle='--')
graphics_2.add_line(var_type='_x', var_name='T_R', axis=ax[1], color='black', linestyle='--')
graphics_2.add_line(var_type='_x', var_name='T_K', axis=ax[1], color='red', linestyle='--')
graphics_2.add_line(var_type='_aux', var_name='T_dif', axis=ax[2], color='black', linestyle='--')
graphics_2.add_line(var_type='_u', var_name='Q_dot', axis=ax[3], color='black', linestyle='--')
graphics_2.add_line(var_type='_u', var_name='F', axis=ax[4], color='black', linestyle='--')

ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('T [K]')
ax[2].set_ylabel('$\Delta$ T [K]')
ax[3].set_ylabel('Q [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')

# Custom legend to clarify what the colors mean
# --- CUSTOM LEGEND OVERRIDE ---
# We explicitly define the color and style for the legend handles
custom_handles = [
    Line2D([0], [0], color='tab:blue', lw=2),
    Line2D([0], [0], color='tab:orange', lw=2),
    Line2D([0], [0], color='black', lw=2, linestyle='--'),
    Line2D([0], [0], color='red', lw=2, linestyle='--')
]

ax[0].legend(
    custom_handles, 
    ['C_a (Const)', 'C_b (Const)', 'C_a (Sym)', 'C_b (Sym)'], 
    loc='upper right'
)

fig.align_ylabels()
fig.tight_layout()
plt.ion()

timer_1 = Timer()
timer_2 = Timer()

x0_1 = np.copy(x0)
x0_2 = np.copy(x0)

print("\nStarting Simulation Loop...")
for k in range(50):
    # Step Universe 1
    timer_1.tic()
    u0_1 = mpc_1.make_step(x0_1)
    timer_1.toc()
    y_next_1 = simulator_1.make_step(u0_1)
    x0_1 = estimator_1.make_step(y_next_1)

    # Step Universe 2
    timer_2.tic()
    u0_2 = mpc_2.make_step(x0_2)
    timer_2.toc()
    y_next_2 = simulator_2.make_step(u0_2)
    x0_2 = estimator_2.make_step(y_next_2)

    if show_animation:
        graphics_1.plot_results(t_ind=k)
        graphics_1.plot_predictions(t_ind=k)
        graphics_2.plot_results(t_ind=k)
        graphics_2.plot_predictions(t_ind=k)
        
        graphics_1.reset_axes()
        plt.show()
        plt.pause(0.01)

print("\n==================================")
print("     PERFORMANCE RESULTS          ")
print("==================================")
print("\n--- CONSTANT RTERM ---")
timer_1.info()

print("\n--- SYMBOLIC RTERM ---")
timer_2.info()

input('Press any key to exit.')