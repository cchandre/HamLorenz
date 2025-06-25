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
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.fft import rfft, irfft, rfftfreq, fft, ifft, fftfreq
from scipy.optimize import root_scalar, minimize
from scipy.stats import gaussian_kde, norm, zscore
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeSolution
from scipy.integrate._ivp.ivp import METHODS as IVP_METHODS
from scipy.stats import skew
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pyhamsys import METHODS, HamSys, solve_ivp_symp, solve_ivp_sympext
import warnings
import time
from datetime import date

def integrate_wrapper(args):
    obj, *rest = args
    sol = obj.integrate(*rest)
    return sol.y_events[0]

def ask_for_value(prompt="Select a branch: ", valid_values=None):
    while True:
        try:
            value = int(input(prompt))
            if valid_values is None or value in valid_values:
                return value
            else:
                print(f"Valid options are: {list(valid_values)}")
        except ValueError:
            print("Invalid input. Please enter an integer.")

class HamLorenz:
    def __init__(self, N, K=1, xi=1, f=None, phi=['cubic', 1]): 
        self.N, self.K = N, K
        self.xi = np.asarray(xi)
        if isinstance(xi, (int, float)):
            self.xi = np.full(K, xi)
        elif len(self.xi) >= K:
            self.xi = self.xi[:K]
            warnings.warn('The length of xi should be K. Using the first K values of xi.', UserWarning)
        else:
            self.xi = np.full(K, self.xi[0])
            warnings.warn('The length of xi should be K. Using the first value of xi for all K.', UserWarning)
        x, y = sp.Symbol('x', real=True), sp.Symbol('y', real=True)
        if f is None and isinstance(phi, list) and phi[0] == 'cubic':
            phi_expr, f_expr, invphi_expr = self.cubic_model(b=phi[1])
        elif isinstance(f, sp.Expr) or isinstance(phi, sp.Expr):
            if f is None:
                phi_expr = phi
                f_expr = 1 / sp.diff(phi_expr, x)
            elif phi is None:
                f_expr = f
                phi_expr = sp.integrate(1 / f_expr, x)
            else:
                f_expr, phi_expr = f, phi
            invphi_expr = self.determine_invphi(phi_expr)
        else:
            raise ValueError("The function f and/or phi must be SymPy expressions or phi should be ['cubic', float] cubic.")
        self.phi_matlab = '@(x)' + sp.printing.octave.octave_code(phi_expr)
        df_expr, dphi_expr = sp.diff(f_expr, x), sp.diff(phi_expr, x)
        compatibility_check = sp.simplify(f_expr * sp.diff(phi_expr, x) - 1)
        if compatibility_check != 0:
            raise ValueError('The functions f and phi are not compatible.')
        self.f = sp.lambdify(x, f_expr, modules='numpy')
        self.phi = sp.lambdify(x, phi_expr, modules='numpy')
        self.invphi = sp.lambdify(y, invphi_expr, modules='numpy') if invphi_expr is not None else None
        self.dphi = sp.lambdify(x, dphi_expr, modules='numpy')
        self.df = sp.lambdify(x, df_expr, modules='numpy')
        self._n = np.arange(N)
        self._mstar = [(k - self._n) % (self.K + 1) for k in range(self.K + 1)]
        self._indk = [(self._n % (self.K + 1)) == k for k in range(self.K + 1)]
        kfreq = 2 * np.pi * rfftfreq(self.N)
        self.R = 2 * np.sum(self.xi[:, np.newaxis] * np.sin(np.outer(np.arange(1, K + 1), kfreq)), axis=0)
        self.casimir_coeffs = self.determine_casimirs()
        self.ncasimirs = len(self.casimir_coeffs)  
        self.delta_p, self.delta_n = self._shifts(np.eye(self.N, dtype=int), axis=0)

    def cubic_model(self, b=1):
        x, y = sp.Symbol('x', real=True), sp.Symbol('y', real=True)
        phi = x + b * x**3 / 3
        f = 1 / (1 + b * x**2)
        u = 3 * y / (2 * b) + sp.sqrt(1 / b**3 + (3 * y / (2 * b))**2)
        invphi = u**(1/3) - u**(-1/3) / b
        return phi, f, invphi
    
    def determine_invphi(self, phi):
        x, y = sp.Symbol('x', real=True), sp.Symbol('y', real=True)
        sol = sp.solve(sp.Eq(phi, y), x)
        sol = [s for s in sol if sp.im(s.subs(y, 1)).evalf() == 0]
        if len(sol) == 1:
            return sp.simplify(sol[0])
        elif len(sol) > 1:
            print(f'The inverse of phi has {len(sol)} branches:')
            for i, s in enumerate(sol):
                print(f'    branch {i}: {s}')
            print(f'    numerical inversion: {len(sol)}')
            branch = ask_for_value(valid_values=range(len(sol) + 1))
            return sp.simplify(sol[branch]) if branch < len(sol) else None
        else:
            print("No inverse found.")
            return None

    def determine_casimirs(self):
        def PeriodicKroneckerDelta(i, j):
            return sp.KroneckerDelta(i % self.N, j % self.N)
        J = sp.Matrix(self.N, self.N,\
            lambda n, m: sum(self.xi[k - 1] * (PeriodicKroneckerDelta(n, m - k)\
                                             - PeriodicKroneckerDelta(n, m + k))\
                                                for k in range(1, self.K + 1)))
        J_null = J.nullspace()
        return [np.array(vec.evalf(), dtype=np.float64).reshape(self.N) for vec in J_null] 

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
    
    def _shifts(self, vec, axis=None):
        pshift = np.asarray([np.roll(vec, -k, axis=axis) for k in range(1, self.K + 1)])
        nshift = np.asarray([np.roll(vec, k, axis=axis) for k in range(1, self.K + 1)])
        return pshift, nshift
    
    def l96_dot(self, _, x):
        return np.sum(self.xi[:, np.newaxis] * (np.roll(x, -1) - np.roll(x, 1)), axis=0)
    
    def x_dot(self, _, x):
        fx = self.f(x)
        pshift, nshift = self._shifts(x * fx)
        return fx * np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0)
    
    def y_dot(self, _, y):
        x = self._invphi(y)
        pshift, nshift = self._shifts(x * self.f(x))
        return np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0)
    
    def z_dot(self, _, z):
        x, Q = z[:self.N], z[self.N:].reshape((self.N, self.N))
        dxdt, dQdt = self.x_dot(_, x), self.jacobian(x) @ Q
        return np.concatenate((dxdt, dQdt), axis=None)
    
    def jacobian(self, x):
        fx, dfx = self.f(x), self.df(x)
        pshift, nshift = self._shifts(x * fx)
        diag = np.diag(dfx * np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0))
        pshift, nshift = self._shifts(fx + x * dfx)
        off_diag = np.sum(fx[np.newaxis, :, np.newaxis] * self.xi[:, np.newaxis, np.newaxis]\
              * (pshift[..., np.newaxis] * self.delta_p - nshift[..., np.newaxis] * self.delta_n), axis=0)
        return diag + off_diag
    
    def generate_initial_conditions(self, energy=1, casimirs=0, xmin=None, xmax=None, ntry=5):
        for _ in range(ntry):
            try: 
                rng = np.random.default_rng()
                X = rng.standard_normal(self.N)
                X = np.sqrt(2 * energy) * X / np.linalg.norm(X)
                casimirs = np.atleast_1d(casimirs)
                if len(casimirs) >= self.ncasimirs:
                    casimirs = casimirs[:self.ncasimirs] 
                elif len(casimirs) < self.ncasimirs:
                    casimirs = np.full(self.ncasimirs, casimirs[0])
                cons = [{'type': 'eq', 'fun': lambda x: self.hamiltonian(x) - energy}]
                for k in range(self.ncasimirs):
                    cons.append({'type': 'eq', 'fun': lambda x, k=k: self.casimir(x, k) - casimirs[k]})
                if xmin is not None and xmax is not None:
                    xmin = np.full(self.N, xmin) if np.isscalar(xmin) else np.asarray(xmin)
                    xmax = np.full(self.N, xmax) if np.isscalar(xmax) else np.asarray(xmax)
                    bounds = list(zip(xmin, xmax))
                else:
                    bounds = None
                result = minimize(lambda _: 0, X, constraints=cons, method='SLSQP', bounds=bounds)
                return result.x
            except RuntimeError:
                pass
        raise RuntimeError("Optimization failed: " + result.message)
    
    def integrate(self, tf, x, t_eval=None, events=None, method='RK45', step=1e-2, tol=1e-8):
        start = time.time()
        if method in IVP_METHODS:
            sol = solve_ivp(self.x_dot, (0, tf), x, t_eval=t_eval, events=events, rtol=tol, atol=tol, max_step=step, method=method)
        elif method in METHODS:
            if len(x) % (self.K + 1) == 0:
                sol = solve_ivp_symp(self._chi, self._chi_star, (0, tf), x, t_eval=t_eval, method=method, step=step)
            else:
                hs = HamSys(btype='other')
                hs.coupling, hs.y_dot = self.coupling, self.y_dot
                sol = solve_ivp_sympext(hs, (0, tf), self.phi(x), t_eval=t_eval, method=method, step=step, omega=1)
                sol.y = self._invphi(sol.y)
        else:
            raise ValueError('The chosen method is not valid.')
        print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
        self._compute_error(sol.y, sol.y[:, 0])
        return sol
    
    def _compute_error(self, x, x0):
        energy_init = self.hamiltonian(x0)
        energy_error = np.amax(np.abs(self.hamiltonian(x[:self.N, :]) - energy_init), axis=0)
        print(f'\033[90m        Error in energy = {energy_error:.2e} (initial value = {energy_init:.2e}) \033[00m')
        casimirs_init = [self.casimir(x0, k=k) for k in range(self.ncasimirs)]
        casimirs_error = [np.amax(np.abs(self.casimir(x[:self.N, :], k=k)  - casimirs_init[k]), axis=0) for k in range(self.ncasimirs)]
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
        coeffs = self.casimir_coeffs[k] if x.ndim==1 else self.casimir_coeffs[k][:, np.newaxis]
        return np.sum(self.phi(coeffs * x), axis=0 if x.ndim == 2 else None)
    
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
    
    def coupling(self, h, y, omega=10):
        y1, y2 = np.split(y, 2)
        sy = (y1 + y2) / 2
        dy = irfft(np.exp(2j * omega * h * self.R) * rfft((y1 - y2) / 2, n=len(y1)), n=len(y1)).real
        return np.concatenate((sy + dy, sy - dy), axis=None)
    
    def compute_ps(self, x, tf, ps, dir=1, method='RK45', tol=1e-8, step=1e-2):
        event_func = lambda _, y: ps(y)
        event_func.terminal, event_func.direction = False, dir
        args = [(self, tf, x_, None, event_func, method, step, tol) for x_ in x]
        with Pool() as pool:
            result = pool.map(integrate_wrapper, args)
        return result
    
    def plot_ps(self, vec, indices):
        fig = plt.figure(figsize=(6, 6))
        if len(indices) != 2 and len(indices) != 3:
            raise ValueError('The indices in plot_ps should be of length 2 or 3.')
        if len(indices) == 2:
            ax = fig.add_subplot()
            for x in vec:
                if x.ndim == 2:
                    ax.plot(x[:, indices[0]], x[:, indices[1]], marker='.', markersize=5, linestyle='None')
        else:
            ax = fig.add_subplot(111, projection='3d')
            for x in vec:
                if x.ndim == 2:
                    ax.plot(x[:, indices[0]], x[:, indices[1]], x[:, indices[2]], marker='.', markersize=5, linestyle='None')
            ax.set_zlabel(f'$X_{{{indices[2]}}}$', fontsize=12)
        ax.set_xlabel(f'$X_{{{indices[0]}}}$', fontsize=12)
        ax.set_ylabel(f'$X_{{{indices[1]}}}$', fontsize=12)
        ax.set_title(r'Poincaré section', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def compute_lyapunov(self, tf, x0, reortho_dt, tol=1e-8, method='RK45', plot=True):
        start = time.time()
        lyap_sum = np.zeros(self.N, dtype=np.float64)
        x, Q = x0.copy(), np.eye(self.N, dtype=np.float64)
        for _ in range(int(tf / reortho_dt)):
            z0 = np.concatenate((x, Q), axis=None)
            sol = solve_ivp(self.z_dot, (0, reortho_dt), z0, method=method, t_eval=[reortho_dt], atol=tol, rtol=tol)
            z1 = sol.y[:, -1]
            x, Q = z1[:self.N], z1[self.N:].reshape((self.N, self.N))
            Q, R = np.linalg.qr(Q)
            lyap_sum += np.log(np.abs(np.diag(R)))
        print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
        self._compute_error(x[:, np.newaxis], x0)
        lyap_sort = np.sort(lyap_sum / tf)
        print(f'\033[90m        Error in spectrum = {np.amax(np.abs(lyap_sort + lyap_sort[::-1])):.2e} \033[00m')
        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(lyap_sort, linewidth=2)
            plt.xlabel(r'$n$', fontsize=12)
            plt.ylabel(r'$\lambda_n$', fontsize=12)
            plt.title(r'Lyapunov exponents', fontsize=14)
            plt.xlim([1, self.N])
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        return lyap_sort
    
    def skewness(self, sol, n_init=0, var='X'):
        X_t = sol.y[:self.N, n_init:].flatten()
        if var == 'Y':
            Y_t = self.phi(X_t)
        return skew(X_t) if var == 'X' else skew(Y_t)
    
    def desymmetrize(self, vec):
        xf = fft(vec, axis=0)
        phase = np.unwrap(np.angle(xf[1, :]))
        ki = fftfreq(self.N, d=1/self.N)
        return ifft(xf * np.exp(-1j * np.outer(ki, phase)), axis=0).real

    def plot_timeseries(self, sol):
        panel_width, panel_height = 3, 5
        colorbar_width = 0.5
        field, field_sym = sol.y[:self.N, :], self.desymmetrize(sol.y[:self.N, :])
        cmap = 'RdBu_r'
        fig_width = 2 * panel_width + colorbar_width + 1.0
        fig_height = panel_height + 1.0
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, 4, width_ratios=[0.05, 1, 1, 0.05], wspace=0.7)
        cax1 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        cax2 = fig.add_subplot(gs[3])
        im1 = ax1.imshow(field.T, extent=[0, self.N, sol.t[-1], sol.t[0]], cmap=cmap)
        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'Time ($t$)')
        ax1.set_title('Hovmöller diagram')
        im2 = ax2.imshow(field_sym.T, extent=[0, self.N, sol.t[-1], sol.t[0]], cmap=cmap)
        ax2.set_xlabel(r'$n$')
        ax2.set_title('Hovmöller diagram (desymmetrized)')
        ax1.set_aspect('auto')
        ax2.set_aspect('auto')
        cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
        cbar1.set_label('$X_n(t)$')
        cbar1.ax.yaxis.set_label_position('left')
        cbar1.ax.yaxis.tick_left()
        cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical', label='Color scale')
        cbar2.set_label('$X_n(t)$')
        plt.show()

    def plot_pdf(self, sol):
        X_t = zscore(sol.y[:self.N, :].flatten(), ddof=1)
        Y_t = zscore(self.phi(sol.y[:self.N, :].flatten()), ddof=1)
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

    def save2matlab(self, data, filename='data'):
        mdic = {'date': date.today().strftime(' %B %d, %Y'), 'author': 'cristel.chandre@cnrs.fr'}
        mdic.update({'phi': self.phi_matlab, 'N': self.N, 'K': self.K, 'xi': self.xi.tolist()})
        mdic.update({'Casimirs': self.casimir_coeffs})
        if isinstance(data, OdeSolution):
            X = data.y[:self.N, :].T
            Y = self.phi(X)
            mdic.update({'t': data.t, 'X': X, 'Y': Y, 'Xs': self.desymmetrize(X).T, 'Ys': self.desymmetrize(Y).T})
        else:
            mdic.update({'data': data})
        savemat(filename + '.mat', mdic)
        print(f'\033[90m        Results saved in {filename}.mat \033[00m')
