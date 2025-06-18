from hamlorenz import HamLorenz

N= 100

tf = 1e4

energy = 25
casimirs = [24.8, 22.7]

hl = HamLorenz(N)

x0 = hl.generate_initial_conditions(N, energy=energy, casimirs=casimirs)

hl.compute_lyapunov(tf, x0, reortho_dt=10, tol=1e-8, plot=True)
