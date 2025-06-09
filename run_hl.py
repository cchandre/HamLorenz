import sympy as sp
import numpy as np
from hamlorenz import HamLorenz

x = sp.symbols('x')
phi = x + x**3 / 3

N = 100
tf = 5000

hl = HamLorenz(N, phi=phi)

E = 25

x0 = np.random.rand(N)

x0 *= np.sqrt(2 * E / np.sum(x0**2))

print(hl.hamiltonian(x0))

sol = hl.integrate(tf, x0)

hl.plot_timeseries(sol, desymmetrize=True)
