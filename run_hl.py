import sympy as sp
import numpy as np
from hamlorenz import HamLorenz

N = 6
tf = 3e4

hl = HamLorenz(N)

E = 8.074451109489349

a1 = 2.242245751187437

Nt = 50

x0 = [hl.generate_initial_conditions(N, energy=E, casimirs=[2, 18]) for _ in range(Nt)]

result = hl.compute_ps(x0, tf, ps=lambda y: y[5] - a1)

hl.plot_ps(result, indices=(0, 1, 2))

#sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1, with_tangent_flow=True)

#hl.compute_lyapunov(tf, x0, reortho_dt=10, tol=1e-8, plot=True)

#hl.plot_timeseries(sol)

#hl.plot_pdf(sol)

#hl.save2matlab(sol, filename='testdata')
