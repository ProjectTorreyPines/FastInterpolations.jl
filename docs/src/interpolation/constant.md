# Constant Interpolation

Step function interpolation that returns values from neighboring data points without blending.

**Key Feature**: The `side` keyword controls which neighbor to sample.

| Mode | Behavior |
|------|----------|
| `:nearest` | Nearest neighbor — **default** |
| `:left` | Left-continuous (use left neighbor) |
| `:right` | Right-continuous (use right neighbor) |

---

## API Reference

### One-shot (construction + evaluation)

| Function | Description |
|----------|-------------|
| `constant_interp(x, y, xq)` | Constant interpolation at point(s) `xq` |
| `constant_interp(x, y, xq; side=:left)` | With side mode (`:nearest`, `:left`, `:right`) |
| `constant_interp!(out, x, y, xq)` | In-place constant interpolation |
| `constant_interp!(out, x, y, xq; side)` | In-place with side mode |

### Re-usable interpolant

| Function | Description |
|----------|-------------|
| `itp = constant_interp(x, y)` | Create constant interpolant |
| `itp = constant_interp(x, y; side=:left)` | Create with side mode |
| `itp(xq)` | Evaluate at point(s) `xq` |
| `itp(out, xq)` | Evaluate at `xq`, store result in `out` |

### Derivatives

| Function | Description |
|----------|-------------|
| `constant_interp(x, y, xq; deriv=1)` | First derivative (always 0) |
| `deriv1(itp)` | First derivative view |
| `deriv2(itp)` | Second derivative view |

```julia
x = [0.0, 1.0, 2.5, 4.0, 5.0]
y = [1.0, 3.0, 2.0, 4.0, 1.5]
xq = range(x[1], x[end], 100)

# One-shot
constant_interp(x, y, 1.5)              # → 3.0 (nearest)
constant_interp(x, y, 1.5; side=:left)  # → 3.0 (left neighbor)
constant_interp(x, y, 1.5; side=:right) # → 2.0 (right neighbor)

# Interpolant
itp = constant_interp(x, y; side=:nearest)
itp(1.5)                                # evaluate at single point
itp(xq)                                 # evaluate at multiple points

# Derivatives (always 0 for step function)
constant_interp(x, y, 1.5; deriv=1)     # → 0.0
d1 = deriv1(itp); d1(1.5)               # → 0.0
```

!!! note "At Grid Points"
    At exact grid points (`xq == x[i]`), all modes return `y[i]`.

---

## Side Modes Comparison

```@example constant
using FastInterpolations
using Plots

x = [0.0, 1.0, 2.5, 4.0, 5.0]
y = [1.0, 3.0, 2.0, 4.0, 1.5]
xq = range(x[1], x[end], 100)

p = plot(layout=(1,3), size=(900, 280), legend=:topright)
for (i, side) in enumerate([:nearest, :left, :right])
    yq = constant_interp(x, y, xq; side=side)
    plot!(p[i], xq, yq, label="spline", linewidth=2)
    scatter!(p[i], x, y, label="data", markersize=7, color=:black)
    title!(p[i], "side=:$side")
end
p
```

---

## When to Use

- Discrete states or categories
- Sharp transitions must be preserved
- Monotonicity preservation is critical
- Lookup table behavior

**Need smooth curves?** → [Linear](linear.md) or [Cubic](cubic/overview.md)
