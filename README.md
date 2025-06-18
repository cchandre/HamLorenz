## Hamiltonian Lorenz Models

This package implements **Hamiltonian Lorenz-like models**, a class of low-order dynamical systems that extend the classical Lorenz-96 and Lorenz-2005 frameworks by incorporating a **Hamiltonian structure**. These models are designed to preserve certain physical invariants—such as energy and Casimirs—making them particularly well-suited for studying conservative dynamical systems, geophysical flows, and chaotic transport.

### Features

* **Hamiltonian structure**: The time evolution of the system is derived from a Hamiltonian, preserving energy exactly in the continuous-time limit.
* **Casimir invariants**: Multiple conserved quantities beyond energy, ensuring the system evolves on a constrained manifold.
* **Symplectic integrators**: Optional numerical solvers designed for long-time energy preservation.
* **Lyapunov spectrum computation**: Quantifies the level of chaos in the system via Lyapunov exponents.
* **Fourier-based desymmetrization**: Enables symmetry reduction to study physical variables in a more interpretable form.
* **PDF and time series visualization**: Built-in tools to analyze and visualize system statistics and dynamics.

### Applications

* Modeling barotropic dynamics or simplified atmospheric flows.
* Testing chaos detection and prediction techniques.
* Benchmarking conservative integration schemes.

### Reference

For a full mathematical formulation and analysis of these models, see:

**Fedele, Chandre, Horvat, and Žagar**
*Hamiltonian Lorenz-like models*,
*Physica D*, Vol. 472, 134494 (2025).
[https://doi.org/10.1016/j.physd.2024.134494](https://doi.org/10.1016/j.physd.2024.134494)

```bibtex
@article{HamLorenz,
  title = {Hamiltonian Lorenz-like models},
  author = {Francesco Fedele and Cristel Chandre and Martin Horvat and Nedjeljka Žagar},
  journal = {Physica D: Nonlinear Phenomena},
  volume = {472},
  pages = {134494},
  year = {2025},
  doi = {https://doi.org/10.1016/j.physd.2024.134494},
}
```

---

### Documentation & Examples

Examples can be found at [Examples](https://github.com/cchandre/HamLorenz/wiki/Examples)

The full documentation, including detailed function explanations, is available on the [Wiki Page](https://github.com/cchandre/HamLorenz/wiki).

