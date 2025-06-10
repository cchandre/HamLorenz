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
from scipy.stats import gaussian_kde, norm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyhamsys import METHODS, solve_ivp_symp
import warnings
import time

class HamLorenz:
    def __init__(self, N, K=1, xi=1, f=None, phi=None, invphi=None, method='ode45'): 
        self.K, self.N = K, N
        self.method = method
        self.xi = np.asarray(xi)
        if isinstance(xi, (int, float)):
            self.xi = np.full(xi, K)
        elif len(self.xi) != K:
            raise ValueError('The length of xi should be K.')
        if f is None and phi is None:
            raise ValueError('Either f or phi must be provided.')
        x, y = sp.symbols('x y')
        if f is None:
            phi_expr = phi
            f_expr = 1 / sp.diff(phi_expr, x)
        elif phi is None:
            f_expr = f
            phi_expr = f, sp.integrate(1 / f_expr, x)
        else:
            f_expr, phi_expr = f, phi
        dphi_expr = sp.diff(phi_expr, x)
        invphi_expr = invphi
        compatibility_check = sp.simplify(f_expr * sp.diff(phi_expr, x) - 1)
        if compatibility_check != 0:
            raise ValueError('The functions f and phi are not compatible.')
        self.f = sp.lambdify(x, f_expr, modules='numpy')
        self.phi = sp.lambdify(x, phi_expr, modules='numpy')
        self.invphi = None if invphi == None else sp.lambdify(y, invphi_expr, modules='numpy')
        self.dphi = sp.lambdify(x, dphi_expr, modules='numpy')
        self._n = np.arange(N)
        self._mstar = [(k - self._n) % (self.K + 1) for k in range(self.K + 1)]
        self._indk = [(self._n % (self.K + 1)) == k for k in range(self.K + 1)]

    def _invphi(self, x, x0=None):
        if self.invphi is not None:
            return self.invphi(x)
        x = np.asarray(x)
        is_scalar = x.ndim == 0
        if np.isscalar(x0):
            x0s = np.full_like(x, x0, dtype=float)
        else:
            x0s = np.asarray(x0)
            if x0s.shape != x.shape:
                raise ValueError("x0 must be scalar or have the same shape as x")
        def solve_scalar(xi, x0i):
            g = lambda z: self.phi(z) - xi
            return root_scalar(g, x0=x0i, fprime=lambda z: self.dphi(z), method='newton').root
        if is_scalar:
            return solve_scalar(x.item(), x0s.item())
        roots = np.fromiter((solve_scalar(xi, x0i) for xi, x0i in zip(x.flat, x0s.flat)), dtype=float)
        return roots.reshape(x.shape)

    def x_dot(self, _, x):
        pshift = np.asarray([np.roll(x * self.f(x), -k - 1) for k in range(self.K)])
        nshift = np.asarray([np.roll(x * self.f(x), k + 1) for k in range(self.K)])
        return self.f(x) * np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0)
    
    def integrate(self, tf, x, t_eval=None, events=None, method='ode45', step=1e-2, tol=1e-8):
        start = time.time()
        if method == 'ode45':
            sol = solve_ivp(self.x_dot, (0, tf), x, t_eval=t_eval, events=events, rtol=tol, atol=tol, max_step=step)
        elif method in METHODS:
            if len(x) % (self.K + 1) != 0:
                raise ValueError('Symplectic integration can only be done if N is a multiple of K+1.')
            sol = solve_ivp_symp(self._chi, self._chi_star, (0, tf), x, t_eval=t_eval, method=method, step=step)
        else:
            raise ValueError('The chosen method is not valid.')
        
        print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
        self._compute_error(sol)
        return sol
    
    def _compute_error(self, sol):
        energy_init = self.hamiltonian(sol.y[:, 0])
        energy_error = np.amax(np.abs(self.hamiltonian(sol.y) - energy_init), axis=0)
        print(f'\033[90m        Error in energy = {energy_error:.2e} (initial value = {energy_init:.2e}) \033[00m')
        casimirs_init = self.casimir(sol.y[:, 0])
        casimirs_error = np.amax(np.abs(self.casimir(sol.y)  - casimirs_init[:, np.newaxis]), axis=1)
        for _, (cas, err) in enumerate(zip(casimirs_init, casimirs_error)):
            print(f'\033[90m        Error in Casimir invariant {_} = {err:.2e} (initial value = {cas:.2e}) \033[00m')
    
    def _kappa(self, k, x):
        mstar, indk = self._mstar[k], self._indk[k]==0
        pshift = (self._n + mstar) % (len(x))
        nshift = (self._n + mstar - self.K - 1) % (len(x))
        kappa = np.zeros_like(x)
        kappa[indk] = self.xi[mstar[indk] - 1] * x[pshift[indk]] * self.f(x[pshift[indk]])\
              - self.xi[-mstar[indk] + 1] * x[nshift[indk]] * self.f(x[nshift[indk]])
        return kappa
    
    def casimir(self, x):
        if np.array_equal(self.xi, self.xi[::-1]) and len(x) % (self.K + 1) == 0:
            return np.asarray([np.sum(self.phi(x[self._indk[k]]), axis=0) for k in range(self.K + 1)])
        return np.asarray([np.sum(self.phi(x), axis=0)])
    
    def hamiltonian(self, x):
        return np.sum(x**2, axis=0) / 2
    
    def _mapk(self, k, x, h):
        indk = self._indk[k]==0
        kappa = self._kappa(k, x)
        x[indk] = self._invphi(kappa[indk] * h + self.phi(x[indk]), x0=self.phi(x[indk]))
        return x
    
    def _chi(self, h, _, x):
        for k in range(self.K + 1):
            x = self._mapk(k, x, h)
        return x
    
    def _chi_star(self, h, _, x):
        for k in reversed(range(self.K + 1)):
            x = self._mapk(k, x, h)
        return x
    
    def desymmetrize(self, sol):
        xf = fft(sol.y, axis=0)
        phase = np.unwrap(np.angle(xf[1, :]))
        ki = 2 * np.pi * fftfreq(self.N)
        return ifft(xf * np.exp(-1j * np.outer(ki, phase)), axis=0).real

    def plot_timeseries(self, sol, desymmetrize=False):
        field = sol.y if not desymmetrize else self.desymmetrize(sol)
        plt.figure(figsize=(10, 5))
        im = plt.imshow(field.T, extent=[0, self.N, sol.t[-1], sol.t[0]], aspect='auto', cmap='RdBu_r', interpolation='none')
        plt.xlabel(r'$n$')
        plt.ylabel(r'Time ($t$)')
        title = 'Hovmöller Diagram'
        if desymmetrize:
            title += ' (desymmetrized)'
        plt.title(title)
        plt.colorbar(im, label=r'$X_n(t)$')
        plt.tight_layout()
        plt.show()

    def plot_pdf(self, sol):
        X_t = sol.y.flatten()
        kde = gaussian_kde(X_t)
        x_vals = np.linspace(min(X_t), max(X_t), 200)
        pdf_kde = kde(x_vals)
        mu, sigma = norm.fit(X_t)
        pdf_gauss = norm.pdf(x_vals, mu, sigma)
        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, pdf_kde, label='KDE estimate', linewidth=2)
        plt.plot(x_vals, pdf_gauss, 'r--', label=fr'Gaussian fit: $\mu={mu:.2f}$, $\sigma={sigma:.2f}$')
        plt.yscale('log')
        plt.xlabel(r'$X$', fontsize=12)
        plt.ylabel(r'PDF', fontsize=12)
        plt.title(r'PDF of $X$ with Gaussian Fit', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
        
        
                      

