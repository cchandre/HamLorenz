from hamlorenz import HamLorenz
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from skimage.measure import euler_number

# Flag: compute Lyapunov exponents (0 = off, 1 = on)
LYAPUNOV = 0

# --- Solve cubic Hamiltonian Lorenz model ---

N = 100              # Number of spatial nodes
tf = 1e3             # Final integration time
energy = 15          # Initial energy
casimirs = [10, 12]  # Initial Casimir constraints

# Define symbolic potential φ(x) = x + a x² + b x³
x = sp.symbols('x', real=True)
mu = 1.5
a = mu / 2
b = 3 * mu**2 / 8
phi = x + a * x**2 + b * x**3

# Create model instance
hl = HamLorenz(N, phi=phi)

# Generate initial conditions with given energy and Casimirs
x0 = hl.generate_initial_conditions(N, energy=energy, casimirs=casimirs)

# Integrate the system using BM4 method
sol = hl.integrate(tf, x0, t_eval=np.arange(tf), method='BM4', step=1e-1)

# Plot time series of the solution
hl.plot_timeseries(sol)

# Save solution to MATLAB format
hl.save2matlab(sol, filename='Xfield')

# Optional: Compute Lyapunov exponents
if LYAPUNOV == 1:
    hl.compute_lyapunov(tf, x0, reortho_dt=10, tol=1e-8, plot=True)

# --- Statistical analysis ---

Xs = sol.y                   # Extract X field (Nx x NT)
Ys = Xs + a * Xs**2 + b * Xs**3  # Nonlinear transformation Y = φ(X)

# Normalize both fields to zero mean and unit variance
Xs = (Xs - np.mean(Xs)) / np.std(Xs)
Ys = (Ys - np.mean(Ys)) / np.std(Ys)

# Determine min/max thresholds from Ys
hmin, hmax = np.min(Ys), np.max(Ys)
h = 0.5 * hmax  # Set excursion threshold

# Compute excursion set Ys > h
excursion = Ys > h

# Plot original Y field and excursion set
plt.figure(figsize=(6, 6))

plt.subplot(1, 2, 1)
plt.imshow(Ys.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Y')
plt.title('Y field')
plt.ylabel('Time')
plt.xlabel('Space')

plt.subplot(1, 2, 2)
plt.imshow(excursion.T, aspect='auto', origin='lower', cmap='gray_r')
plt.title(f'Excursion Set: Y > {h:.2f}')
plt.ylabel('Time')
plt.xlabel('Space')

plt.tight_layout()
plt.show()

# Initialize Euler characteristic and probability arrays
M = 50
dz = (hmax - hmin) / M
zv = np.arange(hmin + dz / 2, hmax, dz)
EC = np.zeros_like(zv)    # For positive/negative peaks
Pz = np.zeros_like(zv)
E1C = np.zeros_like(zv)   # For spacetime excursion set
P1z = np.zeros_like(zv)

# Compute EC and probability for both positive and negative thresholds
for j, zh in enumerate(zv):
    if zh > 0:
        binary_field = Ys > zh
        EC[j] = euler_number(binary_field, connectivity=2)
        Pz[j] = np.sum(Ys > zh)
    elif zh < 0:
        binary_field = np.abs(Ys) > -zh
        EC[j] = euler_number(binary_field, connectivity=2)
        Pz[j] = np.sum(np.abs(Ys) > -zh)

Pz /= Ys.size  # Normalize probability

# Compute EC and probability for spacetime field (Ys > zh only)
for j, zh in enumerate(zv):
    binary_field = Ys > zh
    E1C[j] = euler_number(binary_field, connectivity=2)
    P1z[j] = np.sum(Ys > zh)

P1z /= Ys.size

# Gaussian (Rayleigh-like) reference distribution
pdfR = np.exp(-zv**2 / 2)

# Compute PDFs for Xs, Ys and a Gaussian fit
bins = 50
pdfX, bin_edges = np.histogram(Xs.flatten(), bins=bins, density=True)
Xga = (bin_edges[:-1] + bin_edges[1:]) / 2
pdfG = (1 / np.sqrt(2 * np.pi)) * np.exp(-Xga**2 / 2)

pdfY, bin_edges = np.histogram(Ys.flatten(), bins=bins, density=True)
Yga = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute kurtosis for X and Y fields
kurtX = np.mean((Xs.flatten() - np.mean(Xs))**4) / np.var(Xs)**2
kurtY = np.mean((Ys.flatten() - np.mean(Ys))**4) / np.var(Ys)**2

# --- Plotting ---

# PDF comparison plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.semilogy(Xga, pdfX, marker='s', markersize=3, color='blue', linestyle='None', label='X field')
plt.semilogy(Yga, pdfY, marker='s', markersize=3, color='green', linestyle='None', label='Y field')
plt.semilogy(Xga, pdfG, label='Gauss')
plt.title(f'kurtosis X = {kurtX:.2f}, kurtosis Y = {kurtY:.2f}')
plt.xlabel('X and Y')
plt.ylabel('Probability density (pdf)')
plt.ylim(1e-5, 1)
plt.legend()

# Semilog plot of Euler characteristics
plt.subplot(1, 2, 2)
plt.semilogy(zv, np.abs(EC), marker='s', markersize=3, color='blue', linestyle='None', label='Y field')
plt.semilogy(zv, np.exp(0.5) * np.max(np.abs(EC)) * np.abs(zv) * np.exp(-zv**2 / 2), 'r', label='Gauss')
plt.xlabel('Threshold z')
plt.ylabel('EC(z)')
plt.ylim(1e-1, 2 * np.max(EC))
plt.title('Euler Characteristics: Positive/Negative Peaks')
plt.legend()

# Linear plot of Euler characteristic (spacetime)
plt.figure(figsize=(5, 5))
plt.plot(zv, E1C, marker='s', markersize=3, color='blue', linestyle='--', label='Y field')
plt.plot(zv, np.exp(0.5) * np.max(np.abs(E1C)) * zv * np.exp(-zv**2 / 2), label='Gauss')
plt.xlabel('Threshold z')
plt.ylabel('EC(z)')
plt.title('Euler Characteristics of the Spacetime Field')
plt.legend()

plt.show()
