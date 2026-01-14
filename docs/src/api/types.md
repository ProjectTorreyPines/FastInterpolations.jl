# Type Reference

## Abstract Types

### Interpolant Hierarchy

```@docs
AbstractInterpolant
AbstractMultiInterpolant
```

## Evaluation Operations

```@docs
AbstractEvalOp
EvalValue
EvalDeriv1
EvalDeriv2
```

## Boundary Conditions

### Cubic Splines (BCPair)

```@docs
AbstractBC
PointBC
Deriv1
Deriv2
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

## Multi-Interpolant Types

### Constant Interpolation

```@docs
ConstantMultiInterpolant
```

### Linear Interpolation

```@docs
LinearMultiInterpolant
```

### Quadratic Interpolation

```@docs
QuadraticMultiInterpolant
```

### Cubic Interpolation

```@docs
CubicMultiInterpolant
precompute_transpose!
```

## Index

```@index
```
