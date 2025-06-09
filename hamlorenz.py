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
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyhamsys import METHODS, solve_ivp_symp

class HamLorenz:
    def __init__(self, N, K=1, xi=1, f=None, phi=None, invphi=None, method='ode45'): 
        self.K, self.N = K, N
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
        self.invphi = invphi
        if invphi is not None:
            is_inv_fg = sp.simplify(phi(invphi(x)) - x) == 0
            is_inv_gf = sp.simplify(invphi(phi(x)) - x) == 0
            if not is_inv_fg and not is_inv_gf:
                self.invphi = None
        compatibility_check = sp.simplify(f_expr * sp.diff(phi_expr, x) - 1)
        if compatibility_check != 0:
            raise ValueError('The functions f and phi are not compatible.')
        self.f = sp.lambdify(x, f_expr, modules='numpy')
        self.phi = sp.lambdify(x, phi_expr, modules='numpy')
        self._n = np.arange(N)
        self._mstar = [(k - self._n) % (self.K + 1) for k in range(self.K + 1)]
        self._indk = [(self._n % (self.K + 1)) == k for k in range(self.K + 1)]

    def _invphi(self, x, x0=None):
        if self.invphi is not None:
            return self.invphi(x)
        x = np.asarray(x)
        is_scalar = x.ndim == 0
        def solve_scalar(xi):
            g = lambda z: self.phi(z) - xi
            return root_scalar(g, x0=x0, method='brentq').root
        if is_scalar:
            return solve_scalar(x.item())
        roots = np.fromiter((solve_scalar(xi) for xi in x.flat), dtype=float)
        return roots.reshape(x.shape)

    def x_dot(self, _, x):
        pshift = [np.roll(x * self.f(x), -k - 1) for k in range(self.K)]
        nshift = [np.roll(x * self.f(x), k + 1) for k in range(self.K)]
        return self.f(x) * np.sum(self.xi * (np.asarray(pshift) - np.asarray(nshift)), axis=0)
    
    def integrate(self, tf, x, t_eval=None, events=None, method='ode45', step=1e-2, tol=1e-8):
        if method == 'ode45':
            return solve_ivp(self.x_dot, (0, tf), x, t_eval=t_eval, events=events, rtol=tol, atol=tol, max_step=step)
        elif method in METHODS:
            if len(x) % (self.K + 1) != 0:
                raise ValueError('Symplectic integration can only be done if N is a multiple of K+1.')
            return solve_ivp_symp(self._chi, self._chi_star, (0, tf), x, t_eval=t_eval, method=method, step=step)
    
    def _kappa(self, k, x):
        mstar, indx = self._mstar[k], self._indk[k]
        pshift = indx + mstar % (len(x))
        nshift = indx + mstar - self.K - 1 % (len(x))
        kappa = np.zeros_like(x)
        kappa[indx] = self.xi[mstar[indx]] * x[pshift] * self.f(x[pshift])\
              - self.xi[-mstar[indx]] * x[nshift] * self.f(x[nshift])
        return kappa
    
    def casimir(self, x):
        if np.array_equal(self.xi, self.xi[::-1]) and len(x) % (self.K + 1) == 0:
            return np.asarray([np.sum(self.phi(x[self._indk[k]])) for k in range(self.K + 1)])
        return np.sum(self.phi(x))
    
    def hamiltonian(self, x):
        return np.sum(x**2) / 2
    
    def poincare(self, tf, arr_x, ps): 
        for x in arr_x:
            sol = self.integrate(tf, x, method='BM4', step=1e-1, t_eval=None)
            tab = ps(sol.y)
    
    def _mapk(self, k, x, h):
        kappa = self._kappa(k, x) 
        y = np.zeros_like(x)
        y[self._indk] = self.invphi(kappa * h + self.phi(x[self._indk]))
        return y
    
    def _chi(self, h, _, x):
        for k in range(self.K + 1):
            x = self._mapk(k, x, h)
        return x
    
    def _chistar(self, h, _, x):
        for k in reversed(range(self.K + 1)):
            x = self._mapk(k, x, h)
        return x
    
    def desymmetrize(self, sol):
        xf = fft(sol.y, axis=0)
        phase = np.angle(xf[1, :])
        ki = 2 * np.pi * fftfreq(self.N)
        return ifft(sol.y * np.exp(-1j * np.outer(ki, phase)), axis=0)

    def plot_timeseries(self, sol, desymmetrize=False):
        field = sol.y if not desymmetrize else self.desymmetrize(sol)
        plt.figure(figsize=(10, 5))
        im = plt.imshow(field, extent=[0, self.N, sol.t[-1], sol.t[0]], aspect='auto', cmap='RdBu_r')
        plt.xlabel('n')
        plt.ylabel('Time (t)')
        title = 'Hovmöller Diagram'
        if desymmetrize:
            title += ' (desymmetrized)'
        plt.title('Hovmöller Diagram')
        plt.colorbar(im, label='Value')
        plt.tight_layout()
        plt.show()

    
        
        
                      

