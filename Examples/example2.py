from hamlorenz import HamLorenz

N = 6
tf = 6e4

hl = HamLorenz(N)

E = 8.074451109489349

a1 = 2.242245751187437

ntraj = 200

x0 = hl.generate_initial_conditions(energy=E, casimirs=[2, 18], ntraj=ntraj)

result = hl.compute_ps(x0, tf, ps=lambda y: y[5] - a1, dir=-1, tol=1e-6, step=1e-1)

hl.save2matlab(result, filename='testdata')

hl.plot_ps(result, indices=(0, 1, 2))