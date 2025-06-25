from hamlorenz import HamLorenz
import numpy as np

N = 100

K = 20

tf = 1e3

hl = HamLorenz(N, K=K, xi=[1/K])

# \phi = x + x^3/3 default 

x0 = hl.generate_initial_conditions( energy=25, casimirs=[24.8, 22.7, 10, 12])

sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1, omega=5)

hl.plot_timeseries(sol)