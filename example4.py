import sympy as sp
import numpy as np
from hamlorenz import HamLorenz

N = 100

tf = 1e3

energy = 15
casimirs = [10, 12]

x = sp.symbol('x')
phi = sp.sinh(x)
f = 1 / sp.cosh(x)
invphi = sp.asinh(x)

hl = HamLorenz(N, f=f, phi=phi, invphi=invphi)

x0 = hl.generate_initial_conditions(N, energy=energy, casimirs=casimirs)

sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1)

hl.plot_timeseries(sol)