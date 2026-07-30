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

The script [`scripts/phs/phs_density_comparison.jl`](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/scripts/phs/phs_density_comparison.jl) demonstrates PHS for **electron density interpolation** in a phenol dimer, recreating Figure 2 in [the paper](https://doi.org/10.1063/5.0090232). It uses:

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
julia --project=scripts scripts/phs/phs_density_comparison.jl
```

To get timings that don't include JIT compilation and stencil caching, run the script twice:

```bash
julia --project=scripts -e 'include("scripts/phs/phs_density_comparison.jl"); include("scripts/phs/phs_density_comparison.jl")'
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
| Nearest            | 5.27e-04 (13740×) | 2.15e+00 (2×) | 1.84e-01 (45×) | 1.34e-01 (970×) |
| Linear             | 1.68e-05 (438×) | 9.34e-01 (1×) | 8.80e-02 (22×) | 2.18e-02 (157×) |
| Cubic              | 4.58e-06 (119×) | 9.73e-01 (1×) | 1.17e-01 (29×) | 3.36e-03 (24×) |
| Cardinal           | 2.29e-05 (597×) | 9.21e-01 (1×) | 9.65e-02 (24×) | 3.47e-03 (25×) |
| PHS                | 3.84e-08 | 1.00e+00 | 4.06e-03 | 1.39e-04 |

#### Gradient Magnitude (|∇ρ|) — Relative Error Statistics (with method-to-PHS ratios)

| Method | Min Error | Max Error | Mean Error | Median Error |
|--------|-----------|-----------|------------|--------------|
| Linear             | 3.75e-05 (56×) | 2.39e+01 (24×) | 4.14e-01 (35×) | 1.89e-01 (173×) |
| Cubic              | 2.39e-05 (36×) | 3.36e+00 (3×) | 3.57e-01 (31×) | 2.65e-02 (24×) |
| Cardinal           | 1.83e-04 (275×) | 2.52e+00 (3×) | 2.48e-01 (21×) | 3.23e-02 (29×) |
| PHS                | 6.65e-07 | 1.00e+00 | 1.17e-02 | 1.10e-03 |

#### Laplacian Magnitude (|∇²ρ|) — Relative Error Statistics (with method-to-PHS ratios)

| Method | Min Error | Max Error | Mean Error | Median Error |
|--------|-----------|-----------|------------|--------------|
| Cubic              | 8.30e-06 (1×) | 1.13e+03 (367×) | 6.41e+00 (138×) | 1.69e-01 (13×) |
| Cardinal           | 7.58e-04 (106×) | 1.96e+02 (64×) | 2.41e+00 (52×) | 5.03e-01 (40×) |
| PHS                | 7.16e-06 | 3.07e+00 | 4.65e-02 | 1.27e-02 |

### Timing Summary (with PHS-to-method ratios) for phenol dimer example

**With optimized `blend_factor=1.0` (default).** The build time was for a 75×113×70 grid, and evaluation times were for 1000 query points along the hydrogen-bond path. Script was run twice to get accurate timings after JIT compilation and stencil caching.

| Method | Build (s) | ρ Time (s) | \|∇ρ\| Time (s) | \|∇²ρ\| Time (s) |
|--------|-----------|------------|----------------|-----------------|
| Nearest            | 0.11987 (11.5×) |   0.02844 (0.1×) |                  — |                    — |
| Linear             | 0.02008 (68.4×) |  0.00008 (22.3×) |     0.03990 (0.2×) |                    — |
| Cubic              |  0.66303 (2.1×) |  0.00013 (13.4×) |    0.00012 (49.7×) |       0.03311 (0.2×) |
| Cardinal           | 0.04289 (32.0×) |   0.00020 (8.4×) |    0.00040 (15.2×) |      0.00039 (20.1×) |
| PHS                |           1.373 |           0.0017 |             0.0061 |               0.0079 |

### Detailed timings (with allocation information)

With optimized `blend_factor=1.0`:

```text
Evaluating along path (1000 points)...
  Density (ρ):
    Nearest ...   0.000027 seconds (0 allocations)
    Linear ...    0.000004 seconds (0 allocations)
    Cubic ...     0.000009 seconds (0 allocations)
    Cardinal ...  0.000002 seconds (0 allocations)
    PHS ...       0.001600 seconds (0 allocations)
  Gradient Magnitude (|∇ρ|):
    Linear ...    0.000224 seconds (13 allocations: 432 bytes)
    Cubic ...     0.000019 seconds (7 allocations: 128 bytes)
    Cardinal ...  0.000413 seconds (7 allocations: 128 bytes)
    PHS ...       0.005900 seconds (7 allocations: 128 bytes)
  Laplacian Magnitude (|∇²ρ|):
    Cubic ...     0.000012 seconds (7 allocations: 336 bytes)
    Cardinal ...  0.000477 seconds (1 allocation: 32 bytes)
    PHS ...       0.009600 seconds (1 allocation: 32 bytes)
```

*PHS achieves much higher accuracy than standard methods, especially for derivatives, with moderate build and evaluation overhead. Optimizations like `blend_factor=1.0` significantly improve performance without sacrificing accuracy for most applications.*

## Performance Tuning and Trade-offs

PHS performance can be tuned using two primary parameters: `blend_factor` and `stencil_size`. Both affect the speed-accuracy trade-off.

### Blend Factor Tuning

The `blend_factor` parameter controls the width of the blending neighborhood. Smaller values use fewer neighboring stencils, reducing computational cost but potentially increasing error. The default is `1.0`, which provides an excellent balance for most applications.

**Quick comparison:**

| blend_factor | Blend Nodes | Build (ms) | Eval (ms) | Max Rel Err | Speedup | Rel.Err |
|---|---|---|---|---|---|---|
| 0.5 | 27 | 11.10 | 0.005 | 1.00e+00 | 4200.63× | 1055140.71× |
| 1.0 | 27 | 10.23 | 3.386 | 1.33e-06 | 6.31× | 1.40× |
| 1.5 | 125 | 15.67 | 3.679 | 8.17e-07 | 5.81× | 0.86× |
| **2.0** | **125** | **13.42** | **21.356** | **9.48e-07** | **baseline** | **1.00×** |

**Key insight:** Values less than 1.0 reduce computational cost but may increase error. Values greater than 1.0 increase accuracy at the expense of computational cost.

For performance profiling and parameter tuning, see the test script:

```bash
julia --project=scripts scripts/phs/blend_factor_test_simple.jl
```

This script measures the performance-accuracy trade-off for different `blend_factor` values on synthetic data.

### Stencil Size Tuning

The `stencil_size` parameter sets the number of nodes per axis in each local stencil. Increasing stencil size improves accuracy but increases cost (scales as stencil_size^N). The default is `8`, which balances accuracy and speed.

**Quick comparison (3D, 40³ grid):**

| stencil_size | Total Coeff | Time(ms) | Max Rel Err | Speedup | Error Ratio |
|---|---|---|---|---|---|
| 3 | 31 | 0.04 | 9.15e-05 | 101.84× | 68.98× |
| 4 | 68 | 0.08 | 8.57e-05 | 43.56× | 64.56× |
| 5 | 129 | 0.26 | 2.95e-05 | 14.35× | 22.22× |
| 6 | 220 | 1.08 | 1.41e-05 | 3.39× | 10.64× |
| 7 | 347 | 2.10 | 1.03e-05 | 1.74× | 7.73× |
| **8** | **516** | **3.67** | **1.33e-06** | **baseline** | **1.00×** |
| 10 | 1004 | 10.02 | 1.32e-06 | 2.73×↓ | 0.99× |

**Key insight:** The default `stencil_size=8` is well-optimized. Smaller sizes (e.g., 6) offer significant speedups but with larger errors. Larger sizes provide diminishing returns on accuracy while increasing cost.

For detailed analysis of stencil size trade-offs, run:

```bash
julia --project=scripts scripts/phs/stencil_size_test.jl
```

This script systematically explores stencil sizes from 3 to 10, measuring performance and accuracy on synthetic data.

### Tuning Recommendations

1. **Default settings** (`stencil_size=8`, `blend_factor=1.0`) are recommended for most applications and provide excellent accuracy-performance balance.

2. **High-accuracy applications** (e.g., quantum chemistry): Keep defaults. Consider `blend_factor=2.0` only if accuracy dominates and 3× longer runtimes are acceptable.

3. **Performance-critical applications** (e.g., real-time approximation): Try `blend_factor=0.5` for ~10-100× speedup with ~50× error increase. Visual accuracy may still be acceptable depending on the application.

4. **Do not reduce `stencil_size` below 8** unless extreme performance is needed. Smaller stencils show significant accuracy degradation and may exhibit convergence issues (stencil_size=3).

5. **Interactive tuning**: Both test scripts generate synthetic 40³ grids for rapid prototyping. For production use, benchmark on realistic grid sizes and data distributions.

### Profiling and Optimization

To identify bottlenecks in your specific use case:

```bash
julia --project=scripts scripts/phs/phs_density_comparison_simplified.jl
```

This script includes profiling infrastructure (50 million sample buffer, thread/task grouping) to visualize which operations consume the most time. Results guide parameter selection:

- **High coefficient evaluation cost** → reduce `blend_factor` or `stencil_size`
- **High Hessian cost** → intrinsic to the algorithm; optimize at the application level
- **Acceptable polynomial overhead** → typically not a tuning target

## References

- **Paper**: [Otero-de-la-Roza, A. *Finding critical points and reconstruction of electron densities on grids*. J. Chem. Phys. 156, 224116 (2022).](https://doi.org/10.1063/5.0090232)
- **critic2**: [Database of PBE all-electron atomic densities](https://github.com/aoterodelaroza/critic2/tree/master/dat/wfc)
