# Boundary Conditions

Boundary conditions (BCs) determine how spline interpolants behave at the domain endpoints. They are essential for **quadratic** and **cubic** splines, which require constraints to uniquely determine the spline coefficients.

!!! note "Not Needed for All Methods"
    - **Constant** and **Linear** interpolation do not require boundary conditions
    - **Quadratic** splines require one BC (single endpoint)
    - **Cubic** splines require two BCs (both endpoints)

---

## Type Hierarchy

```
AbstractBC{T}
├── PointBC{T}              # Single-point BC (shared by quadratic & cubic)
│   ├── Deriv1{T}           # User-specified first derivative
│   ├── Deriv2{T}           # User-specified second derivative
│   ├── Deriv3{T}           # User-specified third derivative
│   └── PolyFit{D,T}        # D-degree polynomial fit (auto-estimated)
│       ├── LinearFit       # = PolyFit{1} (2 points, O(h))
│       ├── QuadraticFit     # = PolyFit{2} (3 points, O(h²))
│       └── CubicFit        # = PolyFit{3} (4 points, O(h³))
├── BCPair{T,L,R}           # Both endpoints (cubic only)
├── PeriodicBC{T}           # Periodic BC (cubic only)
├── ZeroCurvBC{T}            # Zero curvature at both ends (cubic only)
├── ZeroSlopeBC{T}            # Zero slope at both ends (cubic only)
├── MinCurvFit{T}           # Minimum curvature (quadratic only)
├── Left{T,B}               # Wrapper: apply BC at left endpoint
└── Right{T,B}              # Wrapper: apply BC at right endpoint
```

---

## BC Categories

### 1. PointBC (Shared)

`PointBC` types can be used with **both quadratic and cubic** splines. They specify a derivative condition at a single endpoint.

| Type | Meaning | Usage |
|------|---------|-------|
| `Deriv1(v)` | S'(endpoint) = v | Known slope |
| `Deriv2(v)` | S''(endpoint) = v | Known curvature |
| `Deriv3(v)` | S'''(endpoint) = v | Known third derivative |
| `PolyFit{D}()` | Estimate derivative from D+1 points | Auto-fit from data |

**Convenience aliases for PolyFit**:

| Alias | Equivalent | Points | Accuracy |
|-------|------------|--------|----------|
| `LinearFit()` | `PolyFit{1}()` | 2 | O(h) |
| `QuadraticFit()` | `PolyFit{2}()` | 3 | O(h²) |
| `CubicFit()` | `PolyFit{3}()` | 4 | O(h³) |

👉 See [PointBC Details](pointbc.md) for in-depth explanation and visualizations.

### 2. Quadratic-Specific BCs

Quadratic splines need exactly **one** endpoint constraint:

```julia
# Wrap PointBC with Left() or Right() to specify which endpoint
Left(QuadraticFit())     # Default: auto-fit at left endpoint
Right(Deriv1(0.5))      # Slope = 0.5 at right endpoint
Left(Deriv2(0.0))       # Zero curvature at left

# Special quadratic-only BC
MinCurvFit()            # Minimize total curvature (globally smooth)
```

### 3. Cubic-Specific BCs

Cubic splines need constraints at **both** endpoints:

```julia
# BCPair: independent constraints at each endpoint
BCPair(Deriv2(0), Deriv2(0))      # Natural BC
BCPair(Deriv1(0), Deriv1(0))      # Clamped BC
BCPair(Deriv1(1.0), Deriv2(0))    # Mixed

# Convenience shortcuts
ZeroCurvBC()      # = BCPair(Deriv2(0), Deriv2(0))  [default]
ZeroSlopeBC()      # = BCPair(Deriv1(0), Deriv1(0))

# True periodicity (requires y[1] ≈ y[end])
PeriodicBC()     # S(x) = S(x + τ) with C² continuity
```

---

## Quick Reference

### Quadratic Splines

| BC | Description | Best For |
|----|-------------|----------|
| `Left(QuadraticFit())` | 3-point fit at left | **Default** - exact for quadratics |
| `Right(QuadraticFit())` | 3-point fit at right | Right-anchored data |
| `Left(Deriv1(v))` | Known slope at left | Physics constraints |
| `MinCurvFit()` | Minimize curvature | Globally smooth interpolation |

### Cubic Splines

| BC | Description | Best For |
|----|-------------|----------|
| `ZeroCurvBC()` | S''=0 at both ends | **Default** - general data |
| `ZeroSlopeBC()` | S'=0 at both ends | Flat endpoints |
| `BCPair(...)` | Custom at each end | Known derivatives |
| `PeriodicBC()` | True periodicity (inclusive) | Cyclic data with `y[1] ≈ y[end]` |
| `PeriodicBC(endpoint=:exclusive)` | True periodicity (exclusive) | FFT grids, `[0, 2π)` data |
| `CubicFit()` | 4-point polynomial fit | Exact for cubic polynomials |

---

## See Also

- [PointBC Details](pointbc.md) — In-depth explanation of PolyFit with visualizations
- [PeriodicBC Details](periodicbc.md) — Inclusive vs exclusive endpoints, period inference, comparison with `WrapExtrap()`
- [Quadratic Interpolation](../interpolation/quadratic.md) — BC examples in context
- [Cubic Interpolation](../interpolation/cubic.md) — BCPair and PeriodicBC details
