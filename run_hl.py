import sympy as sp
import numpy as np
from hamlorenz import HamLorenz

x = sp.symbols('x')
phi = x + x**3 / 3

x0 = np.asarray([0, 1.1, 2.3, 3.05])

hl = HamLorenz(phi=phi)
print(hl.x_dot(x0))
