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
from scipy.optimize import root_scalar, minimize
from scipy.stats import gaussian_kde, norm, zscore
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from pyhamsys import METHODS, solve_ivp_symp
from scipy.io import savemat
import warnings
import time
from datetime import date

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
        self.ncasimirs = self.K + 1 if np.array_equal(self.xi, self.xi[::-1]) and self.N % (self.K + 1) == 0 else 1

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
    
    def y_dot(self, _, y):
        x = self._invphi(y, 0)
        pshift = np.asarray([np.roll(x * self.f(x), -k - 1) for k in range(self.K)])
        nshift = np.asarray([np.roll(x * self.f(x), k + 1) for k in range(self.K)])
        return np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0)
    
    def generate_initial_conditions(self, N, energy=1, casimirs=0):
        X = 2 * np.random.randn(N) - 1
        X = np.sqrt(2 * energy) * X / np.linalg.norm(X)
        casimirs = np.atleast_1d(casimirs)
        if len(casimirs) != self.ncasimirs:
            casimirs = casimirs[0] * np.ones(self.ncasimirs)
        cons = [{'type': 'eq', 'fun': lambda x: self.hamiltonian(x) - energy}]
        for k in range(self.ncasimirs):
            cons.append({'type': 'eq', 'fun': lambda x, k=k: self.casimir(x, k) - casimirs[k]})
        result = minimize(lambda _: 0, X, constraints=cons, method='SLSQP')
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        return result.x
    
    def integrate(self, tf, x, t_eval=None, events=None, method='ode45', step=1e-2, tol=1e-8):
        start = time.time()
        if method == 'ode45':
            sol = solve_ivp(self.x_dot, (0, tf), x, t_eval=t_eval, events=events, rtol=tol, atol=tol, max_step=step)
        elif method in METHODS:
            if len(x) % (self.K + 1) == 0:
                sol = solve_ivp_symp(self._chi, self._chi_star, (0, tf), x, t_eval=t_eval, method=method, step=step)
            else:
                raise ValueError('Symplectic integration can only be done if N is a multiple of K+1.')
        else:
            raise ValueError('The chosen method is not valid.')
        print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
        self._compute_error(sol)
        return sol
    
    def _compute_error(self, sol):
        energy_init = self.hamiltonian(sol.y[:, 0])
        energy_error = np.amax(np.abs(self.hamiltonian(sol.y) - energy_init), axis=0)
        print(f'\033[90m        Error in energy = {energy_error:.2e} (initial value = {energy_init:.2e}) \033[00m')
        casimirs_init = [self.casimir(sol.y[:, 0], k=k) for k in range(self.ncasimirs)]
        casimirs_error = [np.amax(np.abs(self.casimir(sol.y, k=k)  - casimirs_init[k]), axis=0) for k in range(self.ncasimirs)]
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
    
    def casimir(self, x, k=0):
        if self.ncasimirs >= 2:
            return np.sum(self.phi(x[self._indk[k]]), axis=0)
        return np.sum(self.phi(x), axis=0)
    
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
    
    def desymmetrize(self, vec):
        xf = fft(vec, axis=0)
        phase = np.unwrap(np.angle(xf[1, :]))
        ki = fftfreq(self.N, d=1 / self.N)
        return ifft(xf * np.exp(-1j * np.outer(ki, phase)), axis=0).real

    def plot_timeseries(self, sol):
        panel_width, panel_height = 3, 5
        colorbar_width = 0.3
        field, field_sym = sol.y, self.desymmetrize(sol.y)
        vmin = min(field.min(), field_sym.min())
        vmax = max(field.max(), field_sym.max())
        cmap = 'RdBu_r'
        fig_width = 2 * panel_width + colorbar_width + 1.0
        fig_height = panel_height + 1.0
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])
        im1 = ax1.imshow(field.T, extent=[0, self.N, sol.t[-1], sol.t[0]], vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'Time ($t$)')
        ax1.set_title('Hovmöller diagram')
        im2 = ax2.imshow(field_sym.T, extent=[0, self.N, sol.t[-1], sol.t[0]], vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        ax2.set_xlabel(r'$n$')
        ax2.set_title('Hovmöller diagram (desymmetrized)')
        ax1.set_aspect('auto')
        ax2.set_aspect('auto')
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical', label='Color scale')
        cbar.set_label('$X_n(t)$')
        plt.show()

    def plot_pdf(self, sol):
        X_t = zscore(sol.y.flatten(), ddof=1)
        Y_t = zscore(self.phi(sol.y.flatten()), ddof=1)
        kde_x, kde_y = gaussian_kde(X_t), gaussian_kde(Y_t)
        x_vals, y_vals = np.linspace(min(X_t), max(X_t), 200), np.linspace(min(Y_t), max(Y_t), 200)
        pdf_kde_x, pdf_kde_y = kde_x(x_vals), kde_y(y_vals)
        mu, sigma = norm.fit(X_t)
        pdf_gauss = norm.pdf(x_vals, mu, sigma)
        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, pdf_kde_x, label='KDE estimate of X', linewidth=2)
        plt.plot(y_vals, pdf_kde_y, label='KDE estimate of Y', linewidth=2)
        plt.plot(x_vals, pdf_gauss, 'r--', label=fr'Gaussian fit: $\mu={mu:.2f}$, $\sigma={sigma:.2f}$')
        plt.yscale('log')
        plt.ylim([1e-4, 1])
        plt.xlabel(r'$X$', fontsize=12)
        plt.ylabel(r'PDF', fontsize=12)
        plt.title(r'PDF of $X$ and $Y$ with Gaussian fit', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save2matlab(self, sol, filename='data'):
        mdic = {'date': date.today().strftime(' %B %d, %Y'), 'author': 'cristel.chandre@cnrs.fr'}
        mdic.update({'t': sol.t, 'X': sol.y})
        savemat(filename + '.mat', mdic)
        print(f'\033[90m        Results saved in {filename}.mat \033[00m')
