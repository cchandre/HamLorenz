from hamlorenz import HamLorenz
import numpy as np

N = 100

tf = 1e4

energy = 25
casimirs = [24.8, 22.7]

hl = HamLorenz(N)

x0 = hl.generate_initial_conditions(energy=energy, casimirs=casimirs)

sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1)

hl.plot_pdf(sol)

hl.save2matlab(sol, filename='testdata')