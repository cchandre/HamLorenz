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
from scipy.integrate._ivp.ivp import METHODS as IVP_METHODS
from scipy.stats import skew
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pyhamsys import HamSys, Parameters, METHODS
import warnings
import time
from datetime import date

def integrate_wrapper(args):
    obj, *rest = args
    sol = obj.integrate(*rest, display=False)
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

class HamLorenz(HamSys):
    def __init__(self, N, K=1, xi=1, f=None, phi=['cubic', 1], dissipation=None, forcing=None, diffusion=None):
        super().__init__(ndof=N, btype='other')
        self.N, self.K = N, K
        if np.isscalar(xi):
            self.xi = np.full(K, xi, dtype=float)
        else:
            self.xi = np.asarray(xi, dtype=float)
            if len(self.xi) > K:
                self.xi = self.xi[:K]
                warnings.warn('The length of xi should be K. Using the first K values of xi.', UserWarning)
            elif len(self.xi) < K:
                self.xi = np.full(K, self.xi[0], dtype=float)
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
        compatibility_check = sp.simplify(f_expr * dphi_expr - 1)
        if compatibility_check != 0:
            raise ValueError('The functions f and phi are not compatible.')
        self.f = sp.lambdify(x, f_expr, modules='numpy')
        self.phi = sp.lambdify(x, phi_expr, modules='numpy')
        self.invphi = sp.lambdify(y, invphi_expr, modules='numpy') if invphi_expr is not None else None
        self.dphi = sp.lambdify(x, dphi_expr, modules='numpy')
        self.df = sp.lambdify(x, df_expr, modules='numpy')
        self._n = np.arange(self.N)
        self._mstar = [(k - self._n) % (self.K + 1) for k in range(self.K + 1)]
        self._indk = [(self._n % (self.K + 1)) == k for k in range(self.K + 1)]
        self._active = []
        self._pshift_k = []
        self._nshift_k = []
        self._xi_p = []
        self._xi_n = []
        for k in range(self.K + 1):
            mstar = self._mstar[k]
            active = ~self._indk[k]
            self._active.append(active)
            self._pshift_k.append((self._n[active] + mstar[active]) % self.N)
            self._nshift_k.append((self._n[active] + mstar[active] - self.K - 1) % self.N)
            self._xi_p.append(self.xi[mstar[active] - 1])
            self._xi_n.append(self.xi[self.K - mstar[active]])
        k_vec = np.arange(1, self.K + 1)[:, None]
        self._pshift_all = (self._n[None, :] + k_vec) % self.N
        self._nshift_all = (self._n[None, :] - k_vec) % self.N
        kfreq = 2 * np.pi * rfftfreq(self.N)
        self.lamJ = -2j * np.sum(self.xi[:, np.newaxis] * np.sin(np.outer(np.arange(1, self.K + 1), kfreq)), axis=0)
        self.lam_lap = -4.0 * np.sin(kfreq / 2.0) ** 2
        self.casimir_coeffs = self.determine_casimirs()
        self.ncasimirs = len(self.casimir_coeffs)
        self.delta_p, self.delta_n = self._shifts(np.eye(self.N, dtype=int), axis=0)
        self.dissipation = None if dissipation is None else np.abs(dissipation)
        self.forcing, self.diffusion = forcing, diffusion
        if self.diffusion is not None:
            if np.isscalar(self.diffusion):
                self.diffusion = (float(self.diffusion), 0.0)
            else:
                self.diffusion = (float(self.diffusion[0]), float(self.diffusion[1]) if len(self.diffusion) > 1 else 0.0)
            d2 = self.delta_p[0] - 2 * np.eye(self.N) + self.delta_n[0]
            d4 = d2 @ d2
            self.diff_mat = -self.diffusion[0] * d2 - self.diffusion[1] * d4
        self.J = self.K // 2
        self.w = np.ones(2 * self.J + 1)
        self.w[[0, -1]] = 0.5 + 0.5 * (self.K % 2)
        self._w_over_K = self.w / self.K
        j = np.arange(-self.J, self.J + 1)[:, None]
        self._l2005_minus = (self._n[None, :] - j) % self.N
        self._l2005_plus = (self._n[None, :] + j) % self.N

    def cubic_model(self, b=1):
        x, y = sp.Symbol('x', real=True), sp.Symbol('y', real=True)
        if b == 0:
            return x, 1+ 0 * x, y
        phi = x + b * x**3 / 3
        f = 1 / (1 + b * x**2)
        a = 3 * sp.Abs(y) / (2 * b)
        u = (a + sp.sqrt(1 / b**3 + a**2))**sp.Rational(1, 3)
        invphi = sp.sign(y) * (u - 1 / (b * u))
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

    def determine_casimirs(self, tol=1e-8):
        delta = lambda i, j: sp.KroneckerDelta(i % self.N, j % self.N)
        xi_sp = [sp.Rational(x) for x in self.xi]
        Jsp = sp.Matrix(self.N, self.N,
            lambda n, m: sum(xi_sp[k - 1] * (delta(n, m - k) - delta(n, m + k))
                                                for k in range(1, self.K + 1)))
        Jsp_null = Jsp.nullspace()
        casimirs = [np.array(vec.evalf(), dtype=np.float64).reshape(self.N) for vec in Jsp_null]
        Jnp = np.array(Jsp.evalf(), dtype=np.float64)
        output = []
        for c in casimirs:
            if np.allclose(Jnp @ c, 0, atol=tol): 
                output.append(c)
        return output

    def _invphi(self, x, x0=None, tol=1e-12, maxiter=20):
        if self.invphi is not None:
            return self.invphi(x)
        z = np.array(x if x0 is None else np.broadcast_to(x0, np.shape(x)), dtype=float, copy=True)
        for _ in range(maxiter):
            dz = (self.phi(z) - x) / self.dphi(z)
            z -= dz
            if np.all(np.abs(dz) <= tol * (1.0 + np.abs(z))):
                break
        return z
    
    def _shifts(self, vec, axis=None):
        if axis is None and vec.ndim == 1:
            return vec[self._pshift_all], vec[self._nshift_all]
        pshift = np.asarray([np.roll(vec, -k, axis=axis) for k in range(1, self.K + 1)])
        nshift = np.asarray([np.roll(vec, k, axis=axis) for k in range(1, self.K + 1)])
        return pshift, nshift
    
    def l96_dot(self, _, x):
        dxdt = np.sum(self.xi[:, np.newaxis] * (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1), axis=0)
        if self.dissipation is not None:
            dxdt -= self.dissipation * x
        if self.forcing is not None:
            dxdt += self.forcing
        return dxdt

    def lorenz_bracket(self, X, Y):
        wx = self._w_over_K @ X[self._l2005_minus]
        wy = self._w_over_K @ Y[self._l2005_minus] if Y is not X else wx
        term1 = -np.roll(wx, 2 * self.K) * np.roll(wy, self.K)
        prod = np.roll(wx, self.K) * np.roll(Y, -self.K)
        term2 = self._w_over_K @ prod[self._l2005_plus]
        return term1 + term2

    def l2005_dot(self, _, x):
        dxdt = self.lorenz_bracket(x, x)
        if self.dissipation is not None:
            dxdt -= self.dissipation * x
        if self.forcing is not None:
            dxdt += self.forcing
        return dxdt

    def laplacian(self, x, dx=1.0):
        if x.ndim == 1:
            return (x[self._pshift_all[0]] - 2.0 * x + x[self._nshift_all[0]]) / (dx ** 2)
        return (np.roll(x, -1, axis=0) - 2.0 * x + np.roll(x, 1, axis=0)) / (dx ** 2)

    def biharmonic(self, x, dx=1.0):
        return self.laplacian(self.laplacian(x, dx=dx), dx=dx)

    def _diff_term(self, x):
        d1 = self.laplacian(x)
        d2 = self.biharmonic(x)
        return -self.diffusion[0] * d1 - self.diffusion[1] * d2

    def varx_dot(self, _, x):
        fx = self.f(x)
        pshift, nshift = self._shifts(x * fx)
        dxdt = fx * np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0)
        if self.dissipation is not None:
            dxdt -= self.dissipation * x
        if self.forcing is not None:
            dxdt += self.forcing
        if self.diffusion is not None:
            dxdt += self._diff_term(x)
        return dxdt
    
    def vary_dot(self, _, y):
        x = self._invphi(y)
        fx = self.f(x)
        pshift, nshift = self._shifts(x * fx)
        dxdt = np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0)
        if self.dissipation is not None:
            dxdt -= self.dissipation * x / fx
        if self.forcing is not None:
            dxdt += self.forcing / fx
        if self.diffusion is not None:
            dxdt += self._diff_term(x) / fx
        return dxdt

    def z_dot(self, _, z):
        x, Q = z[:self.N], z[self.N:].reshape((self.N, self.N))
        dxdt, dQdt = self.varx_dot(_, x), self.jacobian(_, x) @ Q
        return np.concatenate((dxdt, dQdt), axis=None)
    
    def jacobian(self, _, x):
        fx = np.broadcast_to(self.f(x), x.shape)
        dfx = np.broadcast_to(self.df(x), x.shape)
        pshift, nshift = self._shifts(x * fx)
        diag = np.diag(dfx * np.sum(self.xi[:, np.newaxis] * (pshift - nshift), axis=0))
        pshift, nshift = self._shifts(fx + x * dfx)
        off_diag = np.sum(fx[np.newaxis, :, np.newaxis] * self.xi[:, np.newaxis, np.newaxis]\
              * (pshift[..., np.newaxis] * self.delta_p - nshift[..., np.newaxis] * self.delta_n), axis=0)
        if self.dissipation is not None:
            diag -= self.dissipation * np.eye(self.N)
        if self.diffusion is not None:
            off_diag += self.diff_mat
        return diag + off_diag
    
    def generate_initial_conditions(self, energy=1, casimirs=0, xmin=None, xmax=None, ntry=5, ntraj=1):
        if ntraj > 1:
            return np.vstack([self.generate_initial_conditions(energy, casimirs, xmin, xmax, ntry) for _ in range(ntraj)])
        rng = np.random.default_rng()
        casimirs = np.atleast_1d(casimirs)
        if len(casimirs) >= self.ncasimirs:
            casimirs = casimirs[:self.ncasimirs]
        else:
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
        for _ in range(ntry):
            X = rng.standard_normal(self.N)
            X = np.sqrt(2 * energy) * X / np.linalg.norm(X)
            result = minimize(lambda _: 0, X, constraints=cons, method='SLSQP', bounds=bounds)
            if result.success:
                return result.x
        raise RuntimeError("Optimization failed: " + result.message)
    
    def integrate(self, tf, x, t_eval=None, events=None, method='BM4', step=None, tol=1e-8, projection='midpoint', coords='x', display=True):
        if step is None:
            step = tf / 100
        if t_eval is None:
            t_eval = np.linspace(0, tf, int(np.round(tf / step)) + 1)
        start = time.time()
        is_y_coords = (coords.lower() == 'y')
        y0 = self.phi(x) if is_y_coords else x
        rhs_func = self.vary_dot if is_y_coords else self.varx_dot
        if method in IVP_METHODS:
            solver_kwargs = dict(method=method, max_step=step, rtol=tol, atol=tol)
            if method in ['Radau', 'BDF', 'LSODA'] and not is_y_coords:
                solver_kwargs['jac'] = self.jacobian
            if events is not None:
                solver_kwargs['events'] = events
            sol = solve_ivp(rhs_func, (0, tf), y0, t_eval=t_eval, **solver_kwargs)
            if is_y_coords:
                sol.y = self._invphi(sol.y)
        elif method in METHODS:
            use_ext = (len(y0) % (self.K + 1) != 0)
            self.y_dot = self.vary_dot if is_y_coords else self.varx_dot
            params = Parameters(
                step=step,
                solver=method,
                extension=use_ext,
                tol=tol,
                projection=projection,
                display=False,
                check_energy=False
            )
            initial_state = y0 if is_y_coords else (self.phi(x) if use_ext else x)
            sol = super().integrate(initial_state, t_eval, params=params)
            if use_ext or is_y_coords:
                sol.y = self._invphi(sol.y)
        else:
            raise ValueError('The chosen method is not valid.')
        if display:
            print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
            self._compute_error(sol.y, sol.y[:, 0])
        return sol
    
    def _compute_error(self, x, x0):
        if self.dissipation is not None or self.forcing is not None or self.diffusion is not None:
            return
        energy_init = self.hamiltonian(x0)
        energy_error = np.amax(np.abs(self.hamiltonian(x[:self.N, :]) - energy_init), axis=0)
        print(f'\033[90m        Error in energy = {energy_error:.2e} (initial value = {energy_init:.2e}) \033[00m')
        casimirs_init = [self.casimir(x0, k=k) for k in range(self.ncasimirs)]
        casimirs_error = [np.amax(np.abs(self.casimir(x[:self.N, :], k=k) - casimirs_init[k]), axis=0) for k in range(self.ncasimirs)]
        for k, (cas, err) in enumerate(zip(casimirs_init, casimirs_error)):
            print(f'\033[90m        Error in Casimir invariant {k} = {err:.2e} (initial value = {cas:.2e}) \033[00m')
    
    def _kappa(self, k, x):
        frozen = self._indk[k]
        xf_frozen = np.empty_like(x)
        xf_frozen[frozen] = x[frozen] * self.f(x[frozen])
        return self._xi_p[k] * xf_frozen[self._pshift_k[k]] - self._xi_n[k] * xf_frozen[self._nshift_k[k]]
    
    def casimir(self, x, k=0):
        return self.casimir_coeffs[k] @ self.phi(x)
    
    def hamiltonian(self, x):
        return np.sum(x**2, axis=0) / 2
    
    def _mapk(self, k, x, h):
        active = self._active[k]
        x_act = x[active]
        kappa_act = self._kappa(k, x)
        x[active] = self._invphi(kappa_act * h + self.phi(x_act), x0=x_act)
        return x
    
    def chi(self, h, _, x):
        x = x.copy()
        for k in range(self.K + 1):
            x = self._mapk(k, x, h)
        if self.diffusion is not None:
            lam = self.lam_lap
            rate = -self.diffusion[0] * lam - self.diffusion[1] * (lam ** 2)
            x = irfft(np.exp(rate * h) * rfft(x, n=self.N), n=self.N).real
        if self.dissipation is not None:
            x *= np.exp(-self.dissipation * h)
        if self.forcing is not None:
            x += self.forcing * h
        return x
    
    def chi_star(self, h, _, x):
        x = x.copy()
        if self.forcing is not None:
            x += self.forcing * h
        if self.diffusion is not None:
            lam = self.lam_lap
            rate = -self.diffusion[0] * lam - self.diffusion[1] * (lam ** 2)
            x = irfft(np.exp(rate * h) * rfft(x, n=self.N), n=self.N).real
        if self.dissipation is not None:
            x *= np.exp(-self.dissipation * h)
        for k in reversed(range(self.K + 1)):
            x = self._mapk(k, x, h)
        return x
    
    def coupling(self, h, y, omega=10):
        y1, y2 = np.split(y, 2)
        sy = (y1 + y2) / 2
        dy = irfft(np.exp(-2 * omega * h * self.lamJ) * rfft((y1 - y2) / 2, n=len(y1)), n=len(y1)).real
        return np.concatenate((sy + dy, sy - dy), axis=None)
    
    def compute_ps(self, x, tf, ps, dir=1, method='RK45', tol=1e-6, step=None):
        x = np.atleast_2d(x)
        event_func = lambda _, y: ps(y)
        event_func.terminal, event_func.direction = False, dir
        args = [(self, tf, x_, [0, tf/2, tf], event_func, method, step, tol) for x_ in x]
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
        nsteps = int(tf / reortho_dt)
        for _ in range(nsteps):
            z0 = np.concatenate((x, Q), axis=None)
            sol = solve_ivp(self.z_dot, (0, reortho_dt), z0, method=method, t_eval=[reortho_dt], atol=tol, rtol=tol)
            z1 = sol.y[:, -1]
            x, Q = z1[:self.N], z1[self.N:].reshape((self.N, self.N))
            Q, R = np.linalg.qr(Q)
            lyap_sum += np.log(np.abs(np.diag(R)))
        print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
        self._compute_error(x[:, np.newaxis], x0)
        lyap_sort = np.sort(lyap_sum / (nsteps * reortho_dt))
        if self.dissipation is None and self.forcing is None and self.diffusion is None:
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
        if hasattr(data, 'y') and hasattr(data, 't'):
            X_raw = data.y[:self.N, :]
            Y_raw = self.phi(X_raw)
            mdic.update({
                't': data.t, 
                'X': X_raw.T, 
                'Y': Y_raw.T, 
                'Xs': self.desymmetrize(X_raw).T, 
                'Ys': self.desymmetrize(Y_raw).T
            })
        else:
            mdic.update({'data': data})
        savemat(filename + '.mat', mdic)
        print(f'\033[90m        Results saved in {filename}.mat \033[00m')
