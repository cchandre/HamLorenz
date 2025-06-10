import sympy as sp
import numpy as np
from hamlorenz import HamLorenz

x, y = sp.symbols('x y')

phi = x + x**3 / 3

a = (3*y + sp.sqrt(4 + 9*y**2))**(1/3)
b = 2**(1/3)
invphi = -b / a + a / b

N = 100
tf = 1000

hl = HamLorenz(N, phi=phi, invphi=invphi)

E = 25

x0 = np.random.rand(N)

x0 *= np.sqrt(2 * E / np.sum(x0**2))

sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-2)

hl.plot_timeseries(sol, desymmetrize=True)
