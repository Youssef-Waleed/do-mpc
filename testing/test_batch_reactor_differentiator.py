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


class TestBatchReactorDiff(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """
        default_path = copy.deepcopy(sys.path)
        sys.path.append('../examples/batch_reactor_differentiator/')
        import template_model
        import template_mpc
        import template_simulator

        self.template_model = reload(template_model)
        self.template_mpc = reload(template_mpc)
        self.template_simulator = reload(template_simulator)
        sys.path = default_path


    def test_SX(self):
        print('Testing SX implementation')
        self.batch_reactor_diff('SX')

    def test_MX(self):
        print('Testing MX implementation')
        self.batch_reactor_diff('MX')

    def batch_reactor_diff(self, symvar_type):
        """
        Get configured do-mpc modules:
        """

        model = self.template_model.template_model(symvar_type)
        mpc = self.template_mpc.template_mpc(model)
        simulator = self.template_simulator.template_simulator(model)
        estimator = do_mpc.estimator.StateFeedback(model)
        """
        Set initial state
        """

        X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
        S_s_0 = 0.5 # This is the controlled variable [mol/l]
        P_s_0 = 0.0 #[C]
        V_s_0 = 120.0 #[C]
        x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

        mpc.x0 = x0
        simulator.x0 = x0
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
        Verify the simulation stepped forward without crashing
        """
        self.assertGreater(mpc.data['_time'][-1], 0.0, "Simulation time did not advance. Solver likely failed.")

        # Store for test reasons
        try:
            do_mpc.data.save_results([mpc, simulator], 'test_save', overwrite=True)
        except:
            raise Exception()


if __name__ == '__main__':
    unittest.main()
