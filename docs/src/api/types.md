# Type Reference

## Abstract Types

### Interpolant Hierarchy

```@docs
AbstractInterpolant
AbstractSeriesInterpolant
```

## Evaluation Operations

```@docs
AbstractEvalOp
EvalValue
EvalDeriv1
EvalDeriv2
EvalDeriv3
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
NaturalBC
ClampedBC
PeriodicBC
```

### Quadratic Splines (Single Endpoint)

```@docs
Left
Right
```

## Search Policies

Search policies control how the interpolant finds the correct interval for a query point. Different policies offer trade-offs between simplicity, performance for sequential queries, and thread safety.

```@docs
AbstractSearchPolicy
Binary
HintedBinary
Linear
LinearBinary
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
