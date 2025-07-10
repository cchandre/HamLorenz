from hamlorenz import HamLorenz
import numpy as np
import sympy as sp

# -------------------------------
# General Simulation Parameters
# -------------------------------
N = 100               # Number of grid points
tf = 1e2              # Final integration time
energy = 15           # Initial energy
casimirs = [10, 12]   # Initial Casimir invariants (even and odd for N even)
LYAPUNOV = 0          # Set to 1 to compute Lyapunov exponents

# -----------------------------------------
# Symbolic variable for defining phi(x)
# -----------------------------------------
x = sp.symbols('x', real=True)

# =====================================================
# SINH MODEL
# Define regularized phi using hyperbolic sine function
# =====================================================
mu = 1.2
alpha = mu / 2
beta = 3 * mu**2 / 8

# Compute sinh model coefficients
a = np.sqrt(6 * beta)
b = np.arctanh((2 * alpha) / a)
c = 1 / (a * np.cosh(b))
d = c * np.sinh(b)

# Define Casimir-generating function φ(x)
phi_sinh = c * sp.sinh(a * x + b) - d

print("\n=== SINH MODEL ===")
print(f"a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, d = {d:.4f}\n")

# Create Hamiltonian Lorenz model and run simulation
hl_sinh = HamLorenz(N, phi=phi_sinh)
x0_sinh = hl_sinh.generate_initial_conditions(energy=energy, casimirs=casimirs)
sol_sinh = hl_sinh.integrate(tf, x0_sinh, t_eval=np.arange(tf), method='BM4', step=0.1)

# Plot time series and PDF
hl_sinh.plot_timeseries(sol_sinh)
hl_sinh.plot_pdf(sol_sinh)

# Optional: Compute Lyapunov exponents
if LYAPUNOV:
    hl_sinh.compute_lyapunov(1e3, x0_sinh, reortho_dt=10, tol=1e-8, plot=True)

# Save results
hl_sinh.save2matlab(sol_sinh, filename='sinhmodel_data')


# =====================================================
# CUBIC MODEL
# Define standard cubic φ(x)
# =====================================================
phi_cubic = x + alpha * x**2 + beta * x**3

print("\n=== CUBIC MODEL ===")
print(f"alpha = {alpha:.4f}, beta = {beta:.4f}\n")

# Create Hamiltonian Lorenz model and run simulation
hl_cubic = HamLorenz(N, phi=phi_cubic)
x0_cubic = hl_cubic.generate_initial_conditions(energy=energy, casimirs=casimirs)
sol_cubic = hl_cubic.integrate(tf, x0_cubic, t_eval=np.arange(tf), method='BM4', step=0.1)

# Plot time series and PDF
hl_cubic.plot_timeseries(sol_cubic)
hl_cubic.plot_pdf(sol_cubic)

# Optional: Compute Lyapunov exponents
if LYAPUNOV:
    hl_cubic.compute_lyapunov(1e3, x0_cubic, reortho_dt=10, tol=1e-8, plot=True)

# Save results
hl_cubic.save2matlab(sol_cubic, filename='cubicmodel_data')
