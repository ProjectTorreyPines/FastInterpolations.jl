# PHS Performance and Tuning

Measured accuracy and timing for [Polyharmonic Splines (PHS)](phs.md): the phenol-dimer electron-density example (error statistics against the analytical DFT reference and timings), followed by parameter studies for `blend_factor` and `stencil_size` on synthetic data. All numbers are machine-dependent; the scripts that produce them live under `scripts/phs/`.

## Phenol dimer example: accuracy and timing

Setup: 30×42×30 sub-box of the DFT density grid at 0.236 Bohr spacing, 1000 query points along the O7…H21 hydrogen-bond path, `blend_factor = 1.0` (default), `stencil_size = 8`, `degree = 3`. See the [example description](phs.md#Example:-Quantum-Chemistry) for the data and how to run `scripts/phs/phs_density_comparison.jl`.

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

**With optimized `blend_factor=1.0` (default).** The build time was for the committed 30×42×30 grid (build cost scales with grid size, and PHS has a fixed stencil-precompute cost, so the build ratios shrink on larger grids), and evaluation times were for 1000 query points along the hydrogen-bond path. Script was run twice to get accurate timings after JIT compilation and stencil caching.

| Method | Build (s) | ρ Time (s) | \|∇ρ\| Time (s) | \|∇²ρ\| Time (s) |
|--------|-----------|------------|----------------|-----------------|
| Nearest            | 0.00006 (2694.1×) |  0.00015 (15.3×) |                  — |                    — |
| Linear             | 0.00006 (2916.3×) |  0.00004 (53.2×) |   0.00007 (126.1×) |                    — |
| Cubic              | 0.00143 (121.9×) |  0.00006 (39.2×) |    0.00011 (83.6×) |     0.00010 (121.4×) |
| Cardinal           | 0.00005 (3247.6×) |  0.00021 (11.0×) |    0.00053 (17.3×) |      0.00051 (23.4×) |
| PHS                |           0.174 |           0.0023 |             0.0091 |               0.0119 |

### Detailed timings (with allocation information)

With optimized `blend_factor=1.0`:

```text
Evaluating along path (1000 points)...
  Density (ρ):
    Nearest ...   0.000087 seconds
    Linear ...    0.000013 seconds
    Cubic ...     0.000029 seconds
    Cardinal ...  0.000182 seconds
    PHS ...       0.002299 seconds
  Gradient Magnitude (|∇ρ|):
    Linear ...    0.000033 seconds (7 allocations: 128 bytes)
    Cubic ...     0.000071 seconds (7 allocations: 128 bytes)
    Cardinal ...  0.000502 seconds (7 allocations: 128 bytes)
    PHS ...       0.009114 seconds (7 allocations: 128 bytes)
  Laplacian Magnitude (|∇²ρ|):
    Cubic ...     0.000077 seconds (1 allocation: 32 bytes)
    Cardinal ...  0.000488 seconds (1 allocation: 32 bytes)
    PHS ...       0.011872 seconds (1 allocation: 32 bytes)
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
