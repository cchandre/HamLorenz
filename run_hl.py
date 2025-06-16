import sympy as sp
import numpy as np
from hamlorenz import HamLorenz

N = 100
tf = 1000

hl = HamLorenz(N)

E = 25

x0 = hl.generate_initial_conditions(N, energy=E, casimirs=[24.8, 22.7])

#sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1, with_tangent_flow=True)

hl.compute_lyapunov(tf, x0, reortho_dt=10, tol=1e-8, plot=True)

#hl.plot_timeseries(sol)

#hl.plot_pdf(sol)

#hl.save2matlab(sol, filename='testdata')
