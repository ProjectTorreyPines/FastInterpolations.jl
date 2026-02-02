# Automatic Differentiation Support

FastInterpolations.jl supports multiple AD (Automatic Differentiation) backends, enabling seamless integration with optimization and machine learning workflows.

## When to Use AD vs Built-in Derivatives

For **simple derivatives** of the interpolant itself, use the built-in `deriv` keyword—it's faster and zero-allocation. See [Derivatives](../interpolation/derivatives.md) for details.

**Use AD when** the interpolant is part of a larger differentiable pipeline:
- Optimization: `minimize(p -> loss(itp, p), params)`
- Composed functions: `gradient(x -> sin(itp(x)) + exp(itp(x)), x0)`
- Neural network integration (Flux.jl, Lux.jl)
- Sensitivity analysis in differential equations

## Supported AD Backends

| Backend | Mode | Notes |
|---------|------|-------|
| **ForwardDiff.jl** | Forward | Best for few-input, many-output |
| **Zygote.jl** | Reverse | Best for many-input, few-output |
| **Enzyme.jl** | LLVM-level | Fastest, most restrictive |

### Support Matrix

| Feature | ForwardDiff | Zygote | Enzyme |
|:--------|:-----------:|:------:|:------:|
| Constant | ✅ | ✅ | ✅ |
| Linear | ✅ | ✅ | ✅ |
| Quadratic | ✅ | ✅ | ✅ |
| Cubic | ✅ | ✅ | ✅ |
| Series | ✅ | — | — |
| One-shot | ✅ | ✅ | Linear |
| Complex | ✅ | `real()`/`imag()` | `real()`/`imag()` |

## ForwardDiff.jl (Recommended)

ForwardDiff works seamlessly with all APIs and value types:

```@example ad
using ForwardDiff, FastInterpolations

x = 0.0:0.5:5.0
y = sin.(x)
itp = cubic_interp(x, y; extrap=:extension)

# Compute derivative via AD
grad = ForwardDiff.derivative(itp, 1.5)
nothing #hide
```

```@example ad
# Works with one-shot API too
grad_oneshot = ForwardDiff.derivative(q -> cubic_interp(x, y, q), 1.5)
nothing #hide
```

```@example ad
# Series interpolant support
y1, y2 = sin.(x), cos.(x)
sitp = cubic_interp(x, [y1, y2]; extrap=:extension)
grad_series = ForwardDiff.derivative(q -> sum(sitp(q)), 1.5)
nothing #hide
```

```@example ad
# Vector query: use broadcast + gradient (not derivative)
xq = [0.5, 1.0, 1.5]
grads = ForwardDiff.gradient(v -> sum(itp.(v)), xq)
nothing #hide
```

**Supported:**
- Single interpolants (all types)
- Series interpolants
- One-shot API
- Real and Complex values

## Zygote.jl

Zygote uses source-to-source transformation for reverse-mode AD.

```@example zygote
using Zygote, FastInterpolations

# Note: Use collect() to convert range to Vector for Zygote compatibility
x = collect(0.0:0.5:5.0)
y = 2.0 .* x .+ 1.0
itp = linear_interp(x, y; extrap=:extension)

# Single interpolant works
grad = Zygote.gradient(itp, 1.5)[1]  # Returns 2.0
nothing #hide
```

**Limitations:**
- **Coordinate arrays**: Must use `Vector` (use `collect(range)` instead of `range`)
- **Complex output**: Must use `real()` or `imag()` wrapper
  ```@example zygote
  y_complex = (2.0 + 1.0im) .* x
  itp_complex = linear_interp(x, y_complex; extrap=:extension)

  # ❌ Zygote.gradient(itp_complex, 1.5) fails on complex output
  # ✅ Use this instead:
  grad_real = Zygote.gradient(q -> real(itp_complex(q)), 1.5)[1]
  grad_imag = Zygote.gradient(q -> imag(itp_complex(q)), 1.5)[1]
  nothing #hide
  ```
- **Series interpolants**: Not supported (array mutation)
- **Constant interpolation**: Returns `nothing` instead of `0.0`

## Enzyme.jl

Enzyme performs AD at the LLVM IR level, offering the best performance but with more restrictions.

```@example enzyme
using Enzyme, FastInterpolations

x = 0.0:0.5:5.0
y = x .^ 2
itp = quadratic_interp(x, y; extrap=:extension)

# Use autodiff API
f(xq) = itp(xq)
result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
grad = result[1][1]  # Returns 4.5 (= 2 * 2.25)
nothing #hide
```

**Supported:**
- Single interpolants (Linear, Constant, Quadratic, Cubic)
- Linear one-shot API
- Complex values via `real()`/`imag()`

**Not Supported:**
- Series interpolants (array mutation)
- `Enzyme.gradient` API

