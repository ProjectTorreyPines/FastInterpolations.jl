# Polyharmonic Splines (PHS)

## Overview

Polyharmonic splines (PHS) are **radial basis function (RBF) interpolants** optimized for smooth approximation of multidimensional gridded data. They are particularly effective for **smooth-on-log-scale data** and when combined with **custom reference functions**, enabling physics-informed interpolation in specialized domains like quantum chemistry.

**Key advantages:**
- **C² continuous** with analytical derivatives everywhere
- **Stencil-based evaluation** — cost independent of grid size
- **Log-density transform** — accurate near singularities (e.g., nuclear cusps)
- **Custom reference functions** — encode domain knowledge without modifying the core algorithm

## Mathematical Foundation

This section summarizes the polyharmonic spline method. For full details, see the [paper](https://doi.org/10.1063/5.0090232).

### Basic PHS Interpolant

A polyharmonic spline is constructed as:

$$\omega(x) = \sum_i w_i \phi(\|x - x_i\|) + p(x)$$

where:
- $\{x_i\}$ are $N$ stencil nodes (grid points)
- $\phi(r) = r^K$ is the radial kernel ($K$ odd, typically $K=3$ for $\phi(r)=r^3$)
- $p(x) = v_0 + v_x x + v_y y + v_z z$ is a linear polynomial augmentation
- $w_i$ and $v$ are interpolation coefficients determined by solving:

$$\begin{pmatrix} \Phi & C^T \\ C & 0 \end{pmatrix} \begin{pmatrix} w \\ v \end{pmatrix} = \begin{pmatrix} \rho \\ 0 \end{pmatrix}$$

where $\Phi_{ij} = \phi(\|x_i - x_j\|)$, $C_i = (1, x_i)$, and $\rho = (\rho_1, \ldots, \rho_N)$ are data values.

### Log-Density Smoothing Transform

For data with singularities or rapid variation (e.g., electron density near nuclei), interpolate the transformed function:

$$f(x) = \ln\left(\frac{\rho(x)}{\rho_0(x)}\right)$$

where $\rho_0(x)$ is a smooth **reference function** (e.g., promolecular density, empirical model, or physical constraint).

The interpolant is built on $f(x)$, which is smooth by design. Evaluation recovers the original density and derivatives via the chain rule:

$$\tilde{\rho} = \rho_0 \exp(f)$$

$$\tilde{\rho}_\xi = \rho_0 \left( f_\xi + \frac{\rho_{0\xi}}{\rho_0} \right)$$

$$\tilde{\rho}_{\xi\zeta} = \rho_0 \left( f_{\xi\zeta} + \frac{\rho_\xi \rho_\zeta}{\rho_0^2} + \frac{\rho_{0\xi\zeta}}{\rho_0} - \frac{\rho_{0\xi}\rho_{0\zeta}}{\rho_0^2} \right)$$

### Blending for C² Continuity

Since stencils change discontinuously at grid node boundaries, a **blend function** combines multiple local interpolants:

$$\rho(x) = \frac{\sum_i w_i(x) \tilde{\rho}_i(x)}{\sum_i w_i(x)}$$

with smooth weight $w_i(x)$ that transitions from 1 at node $x_i$ to 0 at distance $a$ (blend range). This ensures $C^2$ continuity across the domain.

## API Usage

### Basic PHS Interpolation

```julia
using FastInterpolations

# Define grid and data
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 20)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# Create interpolant
itp = phs_interp((x, y), data; stencil_size = 8, degree = 3, blend_factor = 1.75)

# Query
val = itp((0.5, 0.3))
grad = itp((0.5, 0.3); deriv = (DerivOp(1), DerivOp(0)))
```

### With Custom Reference Function

```julia
# Define reference density
ref = ConstantRef(1.0)  # simple constant reference

# Build PHS with log-transform
itp = phs_interp((x, y), data; 
    reference_interp = ref, 
    stencil_size = 8, 
    degree = 3)

# Stored data is log(ρ/ρ₀); evaluation returns ρ
val = itp((0.5, 0.3))  # ≈ ρ(0.5, 0.3), not log(ρ)
```

### Custom Reference with Analytical Derivatives

Create a callable reference function supporting the interface `ref(q)` and `ref(q; deriv=(DerivOp(...), ...))`:

```julia
struct MyReference
    # ... state ...
end

(ref::MyReference)(q; deriv=nothing) = begin
    if deriv === nothing
        # return value
        return ρ₀(q)
    else
        # return derivatives based on deriv tuple
        # deriv = (DerivOp(n₁), DerivOp(n₂), DerivOp(n₃))
        # means ∂^(n₁+n₂+n₃)ρ₀/∂x^n₁ ∂y^n₂ ∂z^n₃
        ...
    end
end

itp = phs_interp((x, y, z), rho_data;
    reference_interp = MyReference(),
    stencil_size = 8,
    degree = 3)
```

## Parameters and Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| `stencil_size` | 8 | Stencil nodes per axis (total = stencil_size^N). Increase for smoother but slower interpolant. |
| `degree` | 3 | PHS degree: 1, 3, 5, … (odd only). Higher → smoother, larger condition number. |
| `blend_factor` | 2.0 | Blend range = blend_factor × max_grid_spacing. Increase for wider blending. |
| `reference_interp` | nothing | Optional custom reference function for log-transform. Use `ConstantRef(val)` for constant reference. |
| `reference_data` | nothing | Pre-computed reference values on grid (avoids re-evaluating `reference_interp` at every grid node). |

## Example: Quantum Chemistry

The script [`scripts/phs_density_comparison.jl`](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/feat/phs-interpolation/scripts/phs_density_comparison.jl) demonstrates PHS for **electron density interpolation** in a phenol dimer, recreating Figure 2 in [the paper](https://doi.org/10.1063/5.0090232). It uses:

- **Data**: DFT-computed electron density on a 75×113×70 grid
- **Reference**: Analytical promolecular density (sum of PBE atomic densities from [critic2](https://github.com/aoterodelaroza/critic2))
- **Validation**: Comparison of density, gradient, and Laplacian along a hydrogen-bond path

The resulting plot shows exceptional agreement with analytical values, even near nuclear cusps and steep features:

![PHS density comparison](../images/phs_density_comparison.png)

> **Left column:** Standard 3D interpolation methods (nearest, linear, cubic spline, cardinal) vs. analytical DFT values. All exhibit spurious oscillations and errors near the nuclei. **Right column:** PHS with log-density transform and promolecular reference. Smooth, accurate across the domain, with only minor deviations very close to nuclei.

Polyharmonic spline interpolation was added specifically for applications to physical systems with singularities and steep features, where they achieve better relative results. The results show that PHS with log-density transform and a promolecular reference achieves **orders of magnitude better accuracy** than nearest, linear, cubic spline, and cardinal interpolation for both the density and its derivatives, even near nuclear cusps, at the expense of higher computational cost.

### Running the Example

The script automatically downloads wavefunction files on first run:

```bash
cd scripts/
julia --project=. phs_density_comparison.jl
```

To get timings that don't include JIT compilation and stencil caching, run the script twice:

```bash
cd scripts/
julia --project=. -e 'include("phs_density_comparison.jl"); include("phs_density_comparison.jl")'
```

This generates `phs_density_comparison.png` and demonstrates:

- Loading XYZ atomic geometry
- Building PromolecularRef from critic2 PBE wavefunctions
- Constructing PHS interpolant with log-transform
- Evaluating density, gradient, Laplacian analytically
- Batch evaluation for performance

### Error Statistics (with method-to-PHS ratios) for phenol dimer example

#### Charge Density (ρ) — Relative Error Statistics (with method-to-PHS ratios)

| Method | Min Error | Max Error | Mean Error | Median Error |
|--------|-----------|-----------|------------|--------------|
| Nearest            | 5.27e-04 (35555×) | 2.15e+00 (2×) | 1.84e-01 (45×) | 1.34e-01 (837×) |
| Linear             | 1.68e-05 (1133×) | 9.34e-01 (1×) | 8.80e-02 (22×) | 2.18e-02 (135×) |
| Cubic              | 4.58e-06 (309×) | 9.73e-01 (1×) | 1.17e-01 (29×) | 3.36e-03 (21×) |
| Cardinal           | 2.29e-05 (1546×) | 9.21e-01 (1×) | 9.65e-02 (24×) | 3.47e-03 (22×) |
| PHS                | 1.48e-08 | 1.00e+00 | 4.06e-03 | 1.61e-04 |

#### Gradient Magnitude (|∇ρ|) — Relative Error Statistics (with method-to-PHS ratios)

| Method | Min Error | Max Error | Mean Error | Median Error |
|--------|-----------|-----------|------------|--------------|
| Linear             | 3.75e-05 (49×) | 2.39e+01 (24×) | 4.14e-01 (35×) | 1.89e-01 (171×) |
| Cubic              | 2.39e-05 (31×) | 3.36e+00 (3×) | 3.57e-01 (30×) | 2.65e-02 (24×) |
| Cardinal           | 1.83e-04 (238×) | 2.52e+00 (3×) | 2.48e-01 (21×) | 3.23e-02 (29×) |
| PHS                | 7.68e-07 | 1.00e+00 | 1.17e-02 | 1.11e-03 |

#### Laplacian Magnitude (|∇²ρ|) — Relative Error Statistics (with method-to-PHS ratios)

| Method | Min Error | Max Error | Mean Error | Median Error |
|--------|-----------|-----------|------------|--------------|
| Cubic              | 8.30e-06 (1×) | 1.13e+03 (364×) | 6.41e+00 (135×) | 1.69e-01 (12×) |
| Cardinal           | 7.58e-04 (59×) | 1.96e+02 (63×) | 2.41e+00 (51×) | 5.03e-01 (37×) |
| PHS                | 1.28e-05 | 3.09e+00 | 4.73e-02 | 1.37e-02 |

### Timing Summary (with PHS-to-method ratios) for phenol dimer example

The build time was for a 75×113×70 grid, and evaluation times were for 1000 query points along the hydrogen-bond path. Script was run twice to get accurate timings after JIT compilation and stencil caching.

| Method | Build (s) | ρ Time (s) | \|∇ρ\| Time (s) | \|∇²ρ\| Time (s) |
|--------|-----------|------------|----------------|-----------------|
| Nearest            | 0.00038 (771.4×) |  0.00011 (77.6×) |                  — |                    — |
| Linear             | 0.00028 (1040.4×) |  0.00010 (84.7×) |   0.00019 (169.6×) |                    — |
| Cubic              |  0.03821 (7.6×) |  0.00010 (89.8×) |   0.00014 (233.2×) |     0.00013 (322.4×) |
| Cardinal           | 0.00011 (2599.7×) |  0.00020 (44.2×) |    0.00046 (69.0×) |      0.00045 (93.5×) |
| PHS                |           0.290 |           0.0088 |             0.0319 |               0.0421 |

### Detailed timings (with allocation information)

```text
Evaluating along path (1000 points)...
  Density (ρ):
    Nearest ...   0.000020 seconds (8 allocations: 336 bytes)
    Linear ...    0.000024 seconds (8 allocations: 336 bytes)
    Cubic ...     0.000038 seconds (8 allocations: 336 bytes)
    Cardinal ...  0.000132 seconds (45 allocations: 9.000 KiB)
    PHS ...       0.008705 seconds (0 allocations: 0 bytes)
  Gradient Magnitude (|∇ρ|):
    Linear ...    0.000127 seconds (37 allocations: 1.406 KiB)
    Cubic ...     0.000068 seconds (31 allocations: 1.109 KiB)
    Cardinal ...  0.000391 seconds (142 allocations: 27.125 KiB)
    PHS ...       0.031830 seconds (7 allocations: 128 bytes)
  Laplacian Magnitude (|∇²ρ|):
    Cubic ...     0.000067 seconds (31 allocations: 1.312 KiB)
    Cardinal ...  0.000381 seconds (136 allocations: 27.031 KiB)
    PHS ...       0.042030 seconds (1 allocation: 32 bytes)
```

*PHS is more expensive to build and evaluate than standard methods, but achieves much higher accuracy, especially for derivatives.*

## References

- **Paper**: [Otero-de-la-Roza, A. *Finding critical points and reconstruction of electron densities on grids*. J. Chem. Phys. 156, 224116 (2022).](https://doi.org/10.1063/5.0090232)
- **critic2**: [Database of PBE all-electron atomic densities](https://github.com/aoterodelaroza/critic2/tree/master/dat/wfc)
