import sympy as sp
import numpy as np
from hamlorenz import HamLorenz

N = 100
tf = 1000

hl = HamLorenz(N, K=3, xi=[1/4, 1/2, 1/4])

E = 25

x0 = hl.generate_initial_conditions(N, energy=E, casimirs=[10, 11, 11, 5])

sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1)

hl.plot_timeseries(sol)

hl.plot_pdf(sol)

hl.save2matlab(sol, filename='testdata')
