# Type Reference

## Abstract Types

### Interpolant Hierarchy

```@docs
AbstractInterpolant
AbstractSeriesInterpolant
AbstractInterpolantND
```

### Adjoint Hierarchy

```@docs
AbstractAdjointND
```

### ND Cubic Types

```@docs
AbstractCoeffStrategy
PreCompute
OnTheFly
AutoCoeffs
CubicInterpolantND
```

### Unified API Types

```@docs
HeteroInterpolantND
AbstractInterpMethod
PchipInterp
CardinalInterp
AkimaInterp
CubicHermiteInterp
CubicInterp
LinearInterp
QuadraticInterp
ConstantInterp
NoInterp
GridIdx
interp
interp!
```

### Type Accessors

```@docs
grid_type
value_type
eval_type
```

## Evaluation Operations

```@docs
AbstractEvalOp
DerivOp
deriv_order
EvalValue
EvalDeriv1
EvalDeriv2
EvalDeriv3
```

## Polynomial Coefficients

```@docs
coeffs
CellPoly
```

## Nodal Partials

```@docs
nodal_partials
```

## Vector Calculus (ND)

```@docs
gradient
gradient!
value_gradient
hessian
hessian!
laplacian
```

## Boundary Conditions

### Cubic Splines (BCPair)

```@docs
AbstractBC
PointBC
Deriv1
Deriv2
Deriv3
BCPair
ZeroCurvBC
ZeroSlopeBC
PeriodicBC
```

### PolyFit: Polynomial Fitting BCs

```@docs
PolyFit
LinearFit
QuadraticFit
CubicFit
```

### Endpoint Wrappers (Quadratic Splines)

```@docs
Left
Right
```

## Extrapolation Modes

```@docs
AbstractExtrap
NoExtrap
ClampExtrap
FillExtrap
ExtendExtrap
WrapExtrap
InBounds
ConstExtrap
```

## Search Policies

Search policies control how the interpolant finds the correct interval for a query point.

`AutoSearch` is the **default and recommended choice** for all interpolants. It adapts per-call: `BinarySearch()` for random/scalar queries, `LinearBinarySearch()` for sorted/hinted queries. Most users never need to set a search policy — just pass `hint=Ref(1)` for sequential patterns. See [Search & Hints](@ref search_hints) for details.

```@docs
AbstractSearchPolicy
AutoSearch
BinarySearch
LinearBinarySearch
LinearSearch
```

## Series Input Wrapper

```@docs
Series
n_series
```

## Series Interpolant Types

SeriesInterpolant stores multiple y-series in a **unified matrix** with point-contiguous layout for optimal SIMD performance on scalar queries (10-120× faster than composition-based approaches).

### Constant Interpolation

```@docs
ConstantSeriesInterpolant
```

### Linear Interpolation

```@docs
LinearSeriesInterpolant
```

### Quadratic Interpolation

```@docs
QuadraticSeriesInterpolant
```

### Cubic Interpolation

```@docs
CubicSeriesInterpolant
precompute_transpose!
```

## Index

```@index
```
