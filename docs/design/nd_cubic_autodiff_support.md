# CubicInterpolantND Automatic Differentiation Support

This document describes the AD (Automatic Differentiation) support for `CubicInterpolantND` and the design decisions behind it.

## Overview

`CubicInterpolantND` supports multiple AD approaches with different performance characteristics:

| Method | Time (2D) | Memory | Use Case |
|--------|-----------|--------|----------|
| `deriv` keyword | **47ns** | 80 bytes | Maximum performance |
| ForwardDiff | 414ns | 592 bytes | Standard forward-mode AD |
| Zygote (w/ rrule) | 1.6μs | 6 KiB | Reverse-mode AD |
| Zygote (w/o rrule) | 289μs | 1.49 MiB | Fallback (slow) |

## API Usage

### Direct Analytical Derivatives (Fastest)

```julia
itp = cubic_interp((x, y), data)

# Single partial derivative
∂f_∂x = itp((0.5, 0.5); deriv=(1, 0))
∂f_∂y = itp((0.5, 0.5); deriv=(0, 1))

# Full gradient
grad = [itp(pt; deriv=(1,0)), itp(pt; deriv=(0,1))]
```

### ForwardDiff Integration

```julia
using ForwardDiff

# Vector API (recommended for AD)
itp([0.5, 0.5])  # evaluation
ForwardDiff.gradient(itp, [0.5, 0.5])  # gradient
ForwardDiff.hessian(itp, [0.5, 0.5])   # hessian

# Tuple API also works
ForwardDiff.derivative(q -> itp((q, 0.5)), 0.5)  # partial derivative
```

### Zygote Integration

```julia
using Zygote

# Requires ChainRulesCore (loaded automatically as weakdep)
Zygote.gradient(v -> itp(v), [0.5, 0.5])
```

### Optim.jl Integration

```julia
using Optim

# Direct optimization with AD
result = optimize(itp, x0, LBFGS(); autodiff=:forward)

# Or with analytical gradient for maximum performance
function g!(G, x)
    G[1] = itp((x[1], x[2]); deriv=(1, 0))
    G[2] = itp((x[1], x[2]); deriv=(0, 1))
end
result = optimize(x -> itp((x[1], x[2])), g!, x0, LBFGS())
```

## Implementation Details

### Type Signature Changes for AD

The original tuple constraints `NTuple{N, <:Real}` required homogeneous element types, which fails with ForwardDiff's `Dual` numbers:

```julia
# Problem: Tuple{Dual, Float64} doesn't match NTuple{2, <:Real}
ForwardDiff.gradient(itp, [0.5, 0.5])  # Creates (Dual, Float64) internally
```

**Solution**: Relax to `Tuple{Vararg{Real, N}}` which allows heterogeneous types:

```julia
# Before
query::NTuple{N, <:Real}  # = Tuple{T, T} where T<:Real (same type required)

# After
query::Tuple{Vararg{Real, N}}  # = Tuple{Real, Real} (different types OK)
```

Files modified:
- `src/cubic/nd/nd_eval.jl`: Callable interfaces
- `src/cubic/nd/nd_api.jl`: `cubic_interp` function
- `src/cubic/nd/nd_types.jl`: Helper functions

### Vector Callable API

Added `itp(::AbstractVector)` method for ForwardDiff/Optim.jl compatibility:

```julia
@inline function (itp::CubicInterpolantND{Tg, Tv, N})(
    query::AbstractVector{<:Real}; ...
) where {Tg, Tv, N}
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return itp(query_tuple; ...)
end
```

### ChainRulesCore Extension

Located in `ext/FastInterpolationsChainRulesCoreExt.jl`.

**frule (forward-mode):**
```julia
function ChainRulesCore.frule((_, Δquery), itp::CubicInterpolantND, query)
    y = itp(query)  # primal
    ∂y = sum(Δquery[i] * itp(query; deriv=unit_deriv(i, N)) for i in 1:N)
    return y, ∂y
end
```

**rrule (reverse-mode):**
```julia
function ChainRulesCore.rrule(itp::CubicInterpolantND, query)
    y = itp(query)
    function pullback(Δy)
        ∂query = ntuple(i -> Δy * itp(query; deriv=unit_deriv(i, N)), N)
        return NoTangent(), ∂query
    end
    return y, pullback
end
```

## Performance Analysis

### Why ChainRulesCore Matters for Zygote

Without custom rrule:
- Zygote performs source-to-source transformation
- Traces through all interpolation arithmetic
- Results in massive overhead: **289μs, 1.49 MiB**

With custom rrule:
- Zygote calls our analytical derivative directly
- Uses precomputed Hermite basis derivatives
- Results in: **1.6μs, 6 KiB** (~160x faster)

### ForwardDiff Mechanism

ForwardDiff uses Dual number propagation, not ChainRulesCore:
- Propagates `Dual{T, V, N}` through arithmetic
- Our code preserves Dual types via generic arithmetic
- Performance: **414ns** (fixed, independent of ChainRulesCore)

To potentially speed up ForwardDiff, would need:
1. `ForwardDiffChainRules.jl` bridge (has compatibility issues)
2. Custom Dual number handling (complex, not recommended)

### Analytical vs AD Trade-offs

| Aspect | `deriv` keyword | ForwardDiff | Zygote |
|--------|-----------------|-------------|--------|
| Performance | Best (47ns) | Good (414ns) | OK (1.6μs) |
| Convenience | Manual | Automatic | Automatic |
| Composability | Limited | Full | Full |
| Higher derivatives | Yes (up to 3rd) | Yes | Yes |

**Recommendation:**
- Hot loops, optimization: Use `deriv` keyword
- General use, prototyping: Use ForwardDiff
- Neural networks, Flux: Use Zygote

## Complex-valued Functions

For complex-valued interpolants, `ForwardDiff.gradient` doesn't work because gradient requires scalar output:

```julia
# Error: gradient expects f(x) → Real
ForwardDiff.gradient(itp_complex, [0.5, 0.5])

# Solutions:
# 1. Use jacobian
ForwardDiff.jacobian(v -> [real(itp(v)), imag(itp(v))], [0.5, 0.5])

# 2. Separate real/imag gradients
grad_re = ForwardDiff.gradient(v -> real(itp(v)), [0.5, 0.5])
grad_im = ForwardDiff.gradient(v -> imag(itp(v)), [0.5, 0.5])
```

## Future Improvements

1. **Performance**: Consider `@generated` frule/rrule for compile-time optimization
2. **Enzyme.jl**: Add support when stable for Julia
3. **Batch AD**: Optimize for batched gradient computation
4. **1D/Series interpolants**: Extend ChainRulesCore support to other interpolant types
