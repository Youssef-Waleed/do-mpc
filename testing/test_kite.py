#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import unittest

from importlib import reload
import copy

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc


class TestKite(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """
        default_path = copy.deepcopy(sys.path)
        sys.path.append('../examples/kite/')
        import template_model
        import template_mpc
        import template_simulator

        self.template_model = reload(template_model)
        self.template_mpc = reload(template_mpc)
        self.template_simulator = reload(template_simulator)
        sys.path = default_path


    def test_SX(self):
        print('Testing SX implementation')
        self.kite('SX')

    def test_MX(self):
        print('Testing MX implementation')
        self.kite('MX')

    def kite(self, symvar_type):
        """
        Configure do-mpc modules:
        """
        model = self.template_model.template_model()
        
        np.random.seed(seed=123) # keep the seed constant across tests
 
        w_ref = 6+10*np.random.rand()
        E_0 = 5+3*np.random.rand()
        h_min = 80+40*np.random.rand()  

        # setting up a mpc controller, given the model
        mpc = self.template_mpc.template_mpc(model, w_ref, E_0, h_min=h_min, silence_solver=True)

        # setting up a simulator, given the model
        simulator = self.template_simulator.template_simulator(model, w_ref, E_0)

        # setting up an estimator, given the model
        estimator = do_mpc.estimator.StateFeedback(model)

        """
        Set initial state
        """
        # Derive initial state from bounds:
        lb_theta, ub_theta = mpc.bounds['lower','_x','theta'], mpc.bounds['upper','_x','theta']
        lb_phi, ub_phi = mpc.bounds['lower','_x','phi'], mpc.bounds['upper','_x','phi']
        lb_psi, ub_psi = mpc.bounds['lower','_x','psi'], mpc.bounds['upper','_x','psi']

        # with mean and radius:
        m_theta, r_theta = (ub_theta+lb_theta)/2, (ub_theta-lb_theta)/2
        m_phi, r_phi = (ub_phi+lb_phi)/2, (ub_phi-lb_phi)/2
        m_psi, r_psi = (ub_psi+lb_psi)/2, (ub_psi-lb_psi)/2

        # How close can the intial state be to the bounds?
        # tightness=1 -> Initial state could be on the bounds.
        # tightness=0 -> Initial state will be at the center of the feasible range.

        tightness = 0.6
        theta_0 = m_theta-tightness*r_theta+2*tightness*r_theta*np.random.rand()
        phi_0 = m_phi-tightness*r_phi+2*tightness*r_phi*np.random.rand()
        psi_0 = m_psi-tightness*r_psi+2*tightness*r_psi*np.random.rand()

        # Set the initial state of mpc, simulator and estimator:
        x0 = np.array([theta_0, phi_0, psi_0]).reshape(-1,1)

        # pushing initial condition to mpc, simulator and estimator
        mpc.x0 = x0
        simulator.x0 =x0
        estimator.x0 = x0

        mpc.set_initial_guess()

        """
        Run some steps:
        """

        for k in range(5):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

        """
        Store results (for reference run):
        """
        #do_mpc.data.save_results([mpc, simulator, estimator], 'results_kite', overwrite=True)

        """
        Compare results to reference run:
        """
        ref = do_mpc.data.load_results('./results/results_kite.pkl')

        test = ['_x', '_u', '_time', '_z']

        msg = 'Check if variable {var} for {module} is identical to previous runs: {check}. Max diff is {max_diff:.4E}.'
        for test_i in test:
            # Check MPC
            max_diff = np.max(np.abs(mpc.data.__dict__[test_i] - ref['mpc'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='MPC', check=check, max_diff=max_diff))

            # Check Simulator
            max_diff = np.max(np.abs(simulator.data.__dict__[test_i] - ref['simulator'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='Simulator', check=check, max_diff=max_diff))

            # Estimator
            max_diff = np.max(np.abs(estimator.data.__dict__[test_i] - ref['estimator'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='Estimator', check=check, max_diff=max_diff))


        """
        Store results (from reference run):
        """
        #do_mpc.data.save_results([mpc, simulator, estimator], 'results_kite')

        # Store for test reasons
        try:
            do_mpc.data.save_results([mpc, simulator], 'test_save', overwrite=True)
        except:
            raise Exception()


if __name__ == '__main__':
    unittest.main()
