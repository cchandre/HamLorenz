from hamlorenz import HamLorenz

N = 6

hl = HamLorenz(N)

energy = 8.074451109489349
casimirs = [2, 18]

a1 = 2.242245751187437

tf = 6e4
ntraj = 100

x0 = hl.generate_initial_conditions(energy=energy, casimirs=casimirs, ntraj=ntraj)

result = hl.compute_ps(x0, tf, ps=lambda y: y[5] - a1, dir=-1, tol=1e-8)

hl.save2matlab(result, filename='testdata')

hl.plot_ps(result, indices=(0, 1, 2))