[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://projecttorreypines.github.io/FastInterpolations.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://projecttorreypines.github.io/FastInterpolations.jl/dev/)
[![CI](https://github.com/ProjectTorreyPines/FastInterpolations.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ProjectTorreyPines/FastInterpolations.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/projecttorreypines/fastinterpolations.jl/graph/badge.svg?token=RQ9RwaxeZF)](https://codecov.io/github/projecttorreypines/fastinterpolations.jl)
[![Benchmark](https://img.shields.io/badge/benchmarks-Chart-yellowgreen)](https://projecttorreypines.github.io/FastInterpolations.jl/bench/)

# FastInterpolations.jl

A high-performance 1D interpolation package for Julia, optimized for **zero-allocation hot loops** and **thread-safe** concurrent access.

## Key Strengths

- 🚀 **Fast**: Optimized algorithms that outperform other packages.
- ✅ **Zero-Allocation**: No GC pressure on hot loops.
- 🎯 **Explicit BCs**: Support custom physical boundary conditions.
- 📐 **Derivatives**: Analytical 1st, 2nd, and 3rd derivatives for all methods.
- 🌌 **Generic**: Supports **Complex** values and **AD (AutoDiff)** — ForwardDiff, Zygote, Enzyme.
- 🧵 **Thread-Safe**: Lock-free concurrent access across multiple threads.

## Supported Methods
`FastInterpolations.jl` supports four interpolation methods: `Constant`, `Linear`, `Quadratic`, and `Cubic` splines.

| Method | Continuity | Best For |
|:-------|:-----------|:---------|
| `constant_interp` | C⁻¹ | Step functions (Nearest, Left, Right) |
| `linear_interp` | C⁰ | Simple, fast O(1) range lookup |
| `quadratic_interp` | C¹ | Smooth C¹ continuity with minimal overhead |
| `cubic_interp` | C² | High-quality C² splines (Natural, Clamped, Periodic) |

## Quick Start

`FastInterpolations.jl` provides two primary API styles, plus a specialized **SeriesInterpolant** for multi-series data.

### 1. One-shot API (Dynamic Data)
Best when **`y` values change** every step, but the grid **`x` remains fixed**.

```julia
using FastInterpolations

# Define grid and query points
x = range(0.0, 10.0, 100)   # source grid (100 points)
y = sin.(x)                 # initial y data

xq = range(0.0, 10.0, 500)  # query grid  (500 points)
out = similar(xq)           # pre-allocate output buffer

for t in 1:1000
    @. y = sin(x + 0.01t)           # y values evolve each timestep
    cubic_interp!(out, x, y, xq)    # zero-allocation ✅ (after warm-up)
end
```

### 2. Interpolant API (Static Data)
Best for **fixed lookup tables** where both `x` and `y` are constant.

```julia
x = range(0.0, 10.0, 100)
y = sin.(x)

itp = cubic_interp(x, y)       # pre-compute spline coefficients once

result = itp(5.5)              # evaluate at single point
result = itp(xq)               # evaluate at multiple points
@. result = a * itp(xq) + b    # seamless broadcast fusion
```

### 2.1 SeriesInterpolant (Multiple Series)
When multiple y-series share the same x-grid, use SeriesInterpolant. It leverages **SIMD** and **cache locality** for **10-100× faster** evaluation compared to looping over individual interpolants.

```julia
x = range(0, 10, 100)
y1, y2, y3, y4 = sin.(x), cos.(x), tan.(x), exp.(-x)  # 4 series, same grid

sitp = cubic_interp(x, [y1, y2, y3, y4])   # create SeriesInterpolant
sitp(0.5)  # → 4-element Vector: interpolated values for each series
```

For detailed usage and performance trade-offs, see the [API Selection Guide](https://projecttorreypines.github.io/FastInterpolations.jl/dev/guides/api_selection/).


## Performance

Benchmark comparison against [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl), [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl), and [Dierckx.jl](https://github.com/JuliaMath/Dierckx.jl) for **cubic spline interpolation**.
<!-- BENCHMARK_VERSIONS_START -->
> **Env:** Local · macOS 15.7.3 · Apple M1 Pro · Julia 1.12.4<br>
> **Pkg:** FastInterpolations (v0.2.6) · Interpolations (v0.16.2) · DataInterpolations (v8.9.0) · Dierckx (v0.5.4)
<!-- BENCHMARK_VERSIONS_END -->

![One-Shot](docs/images/benchmark_oneshot_detail.png)

<!-- BENCHMARK_SPEEDUP_START -->
**Speedup:** (2.2 ~ 16.3)× vs `Interpolations.jl` · (9.3 ~ 22.8)× vs `DataInterpolations.jl` · (14.9 ~ 18.3)× vs `Dierckx.jl`
<!-- BENCHMARK_SPEEDUP_END -->

One-shot (construction + evaluation) time per call with fixed grid size $n=100$. `FastInterpolations.jl` is significantly faster even on the first call (cache-miss), and becomes even faster on subsequent calls (cache-hit).

## More Features

```julia
# Analytical derivatives — all methods support 1st, 2nd, 3rd derivatives
cubic_interp(x, y, 5.0; deriv=1)   # 1st derivative at x=5.0
cubic_interp(x, y, 5.0; deriv=2)   # 2nd derivative at x=5.0
cubic_interp(x, y, 5.0; deriv=3)   # 3rd derivative at x=5.0

# Constant interpolation — choose which side to sample
constant_interp(x, y, xq; side=:nearest) # nearest neighbor (default)
constant_interp(x, y, xq; side=:left)    # left-continuous 
constant_interp(x, y, xq; side=:right)   # right-continuous

# Quadratic boundary condition — single endpoint constraint
quadratic_interp(x, y, xq; bc=Left(Deriv1(0.0)))   # S'(left) = 0
quadratic_interp(x, y, xq; bc=Right(Deriv1(1.0)))  # S'(right) = 1

# Cubic boundary conditions — paired endpoint constraints
cubic_interp(x, y, xq; bc=NaturalBC())    # S''=0 at both ends (default)
cubic_interp(x, y, xq; bc=PeriodicBC())   # C²-continuous periodic spline
cubic_interp(x, y, xq; bc=BCPair(Deriv1(2.0), Deriv2(-5.0)))  # custom (left, right) BC
cubic_interp(x, y, xq; bc=CubicFit())     # Estimate derivatives using 4-point fit at both ends 

# Extrapolation modes — all methods support these
linear_interp(x, y, xq; extrap=:constant)    # clamp to boundary values
quadratic_interp(x, y, xq; extrap=:wrap)     # wrap around (periodic data)
cubic_interp(x, y, xq; extrap=:extension)    # extend boundary polynomial
```

## Documentation

For detailed guides on boundary conditions, extrapolation, and performance tuning, visit the [Documentation](https://projecttorreypines.github.io/FastInterpolations.jl).


## License
Apache License 2.0

## Contact
Min-Gu Yoo [![Linkedin](https://i.sstatic.net/gVE0j.png)](https://www.linkedin.com/in/min-gu-yoo-704773230) (General Atomics)  yoom@fusion.gat.com