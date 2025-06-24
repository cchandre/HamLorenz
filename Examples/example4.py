from hamlorenz import HamLorenz
import numpy as np
import sympy as sp

N = 100

tf = 1e3

energy = 15
casimirs = [10, 12]

x = sp.Symbol('x', real=True)
phi = sp.sinh(x)

hl = HamLorenz(N, phi=phi)

x0 = hl.generate_initial_conditions(N, energy=energy, casimirs=casimirs)

sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1)

hl.plot_timeseries(sol)