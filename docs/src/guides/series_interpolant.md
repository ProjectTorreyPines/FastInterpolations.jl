# [SeriesInterpolant Guide](@id series_interpolant)

When multiple y-series share the same x-grid, `SeriesInterpolant` can be **up to ~30× faster** (depending on the number of series and hardware) by leveraging SIMD and cache locality.

## When to Use

SeriesInterpolant is ideal when you have **multiple quantities defined on the same grid**:

| Domain | Example |
|:-------|:--------|
| **ODE/PDE solvers** | Interpolating state variables (position, velocity, acceleration) at adaptive time steps |
| **Physics simulations** | Multiple fields (temperature, pressure, density) on a shared spatial mesh |
| **Financial modeling** | Multiple time series (prices, volumes, indicators) on the same time axis |
| **Signal processing** | Multi-channel sensor data with synchronized sampling |

Instead of creating N separate interpolants and querying them in a loop, SeriesInterpolant evaluates all series in a single vectorized pass.

---

## Creating a SeriesInterpolant

```julia
using FastInterpolations

x = range(0, 10, 100)
y1, y2, y3, y4 = sin.(x), cos.(x), tan.(x), exp.(-x)

# From Vector of Vectors
sitp = cubic_interp(x, [y1, y2, y3, y4])

# From Matrix (columns = series)
Y_matrix = hcat(y1, y2, y3, y4)  # 100×4 matrix
sitp = cubic_interp(x, Y_matrix)
```

!!! note "Flexible Input"
    Any `AbstractMatrix` works — reshape your data as `(n_points × n_series)` and pass it directly.

All interpolation methods support SeriesInterpolant:
- `constant_interp(x, y_series)`
- `linear_interp(x, y_series)`
- `quadratic_interp(x, y_series)`
- `cubic_interp(x, y_series)`

---

## Scalar Query

Evaluate all series at a single point.

```julia
# Allocating (returns new Vector)
vals = sitp(0.5)  # → 4-element Vector{Float64}

# In-place (zero allocation)
output = Vector{Float64}(undef, 4)
sitp(output, 0.5)

# With derivatives
sitp(0.5; deriv=1)        # 1st derivative
sitp(output, 0.5; deriv=2) # 2nd derivative, in-place
```

---

## Vector Query

Evaluate all series at multiple points.

```julia
xq = range(0, 10, 500)

# Allocating (returns Vector of Vectors)
results = sitp(xq)  # [Vector for y1, Vector for y2, ...]

# In-place (zero allocation after warmup)
outputs = [similar(xq) for _ in 1:4]
sitp(outputs, xq)
```

!!! tip "Hot Loop Pattern"
    For hot loops, pre-allocate outputs once outside the loop:
    ```julia
    outputs = [Vector{Float64}(undef, length(xq)) for _ in 1:n_series]

    for t in 1:1000
        # ... update data ...
        sitp(outputs, xq)  # zero allocation
    end
    ```

---

## Performance Tips

### 1. Prefer In-place API

| API | Allocation |
|:----|:-----------|
| `sitp(xq)` | Allocates on every call |
| `sitp(outputs, xq)` | Zero allocation (after warmup) |

### 2. When NOT to use SeriesInterpolant

For very small series counts (n ≤ 2-4) with **vector queries**, a manual loop over individual interpolants may be marginally faster (~10-25%) due to anchor allocation overhead. For **scalar queries** or larger series counts, SeriesInterpolant always wins.

---

## Quick Benchmark

The speedup grows with the number of series — more series means more anchor reuse.

Try this yourself to see the performance difference:

```julia
using FastInterpolations
using BenchmarkTools

# Setup: 100 series on shared x-grid
x = range(0.0, 10.0, 100)
y_series = [n * x.^3 for n in 1:100]

# Baseline: individual interpolants
itps = [cubic_interp(x, y) for y in y_series]

# SeriesInterpolant
sitp = cubic_interp(x, y_series)

# Scalar query comparison (in-place, zero allocation)
out = zeros(length(y_series))

@btime begin
    for (k, itp) in enumerate($itps)
        $out[k] = itp(5.0)
    end
end

@btime $sitp($out, 5.0);
```

Expected output (approximate):
```
  672 ns (0 allocations)    # manual loop
   54 ns (0 allocations)    # SeriesInterpolant ← ~13× faster
```

!!! note "Why SeriesInterpolant is faster"
    Interval lookup and weight computation happen once for all series, then SIMD vectorization and unified matrix storage maximize cache efficiency.

---

## See Also

- [API Selection Guide](api_selection.md) — When to use which API
- [Memory & Allocation](memory_allocation.md) — General optimization patterns
