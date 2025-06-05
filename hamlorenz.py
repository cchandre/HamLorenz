#
# BSD 2-Clause License
#
# Copyright (c) 2025, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyhamsys import METHODS, HamSys, solve_ivp_symp

class HamLorenz:
    def __init__(self, K=1, xi=1, f=None, phi=None, method='ode45'): 
        self.K = K
        self.method = method
        if isinstance(xi, (int, float)):
            self.xi = np.full(xi, K)
        elif len(self.xi) != K:
            raise ValueError('The length of xi should be K.')
        if f is None and phi is None:
            raise ValueError('Either f or phi must be provided.')
        x = sp.symbols('x')
        if f is None:
            phi_expr = phi
            f_expr = 1 / sp.diff(phi_expr, x)
        elif phi is None:
            f_expr = f
            phi_expr = f, sp.integrate(1 / f_expr, x)
        else:
            f_expr, phi_expr = f, phi
        compatibility_check = sp.simplify(f_expr * sp.diff(phi_expr, x) - 1)
        if compatibility_check != 0:
            raise ValueError('The functions f and phi are not compatible.')
        self.f = sp.lambdify(x, f_expr, modules='numpy')
        self.phi = sp.lambdify(x, phi_expr, modules='numpy')

    def x_dot(self, x):
        pshift = [np.roll(x * self.f(x), -k - 1) for k in range(self.K)]
        nshift = [np.roll(x * self.f(x), k + 1) for k in range(self.K)]
        return self.f(x) * np.sum(self.xi * (np.asarray(pshift) - np.asarray(nshift)), axis=0)
    
    def integrate(self, t_eval, x, events=None):
        if self.method == 'ode45':
            return solve_ivp(self.x_dot, (t_eval[0], t_eval[-1]), x, t_eval=t_eval, events=events)
        elif self.method in METHODS:
            if len(x) % (self.K + 1) != 0:
                raise ValueError('Symplectic integration can only be done if N is a multiple of K+1.')
            
    def _mstar(self, l, k):
        return (k - l) % (self.K + 1)
 
    def casimir(self, x):
        return np.sum(self.phi(x))
    
    def hamiltonian(self, x):
        return np.sum(x**2) / 2
    

    

                      

