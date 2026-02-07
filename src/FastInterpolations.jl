module FastInterpolations

import LinearAlgebra
import Printf
using LinearAlgebra: Tridiagonal, lu, ldiv!
using Preferences: @load_preference, @set_preferences!
using AdaptiveArrayPools
using HelpPlots: help_plot, help_plot!

# Module aggregators (encapsulate include order complexity)
include("core/core.jl")
include("linear/linear.jl")
include("constant/constant.jl")
include("quadratic/quadratic.jl")
include("cubic/cubic.jl")

# Derivative view wrapper (depends on all interpolant types)
include("derivative_view.jl")

# Vector calculus operations (depends on ND interpolant types)
include("vector_calculus.jl")

# Custom show methods (depends on all interpolant types and DerivativeView)
include("core/show.jl")

# Exports
export AbstractInterpolant, AbstractSeriesInterpolant, AbstractInterpolantND
export grid_type, value_type, eval_type  # Type introspection for {Tg, Tv} system
export linear_interp, linear_interp!, LinearInterpolant, LinearSeriesInterpolant, LinearInterpolantND
export constant_interp, constant_interp!, ConstantInterpolant, ConstantSeriesInterpolant, ConstantInterpolantND
export quadratic_interp, quadratic_interp!, QuadraticInterpolant, QuadraticSeriesInterpolant, QuadraticInterpolantND
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant, CubicSeriesInterpolant
export CubicInterpolantND, AbstractCoeffStrategy, PreCompute, OnTheFly  # ND cubic types
export gradient, gradient!, hessian, hessian!, laplacian  # Analytical vector calculus for ND
export precompute_transpose!  # Pre-allocate point-contiguous layout for scalar queries
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!

# Search policy types (user-facing algorithm selection)
export AbstractSearchPolicy, Binary, HintedBinary, Linear, LinearBinary

# Boundary condition types
export AbstractBC, PointBC, Deriv1, Deriv2, Deriv3, BCPair
export NaturalBC, ClampedBC, PeriodicBC, MinCurvFit
export PolyFit, LinearFit, QuadraticFit, CubicFit  # Polynomial fitting BCs
export ParabolaFit # Deprecated alias of QuadraticFit
export Left, Right

# Derivative view functions and types
export deriv1, deriv2, deriv3, deriv_view, AbstractDerivativeView, DerivativeView

# Operation types (for derivative dispatch)
export AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3

# Plot recipe helpers (re-exported from HelpPlots)
export help_plot, help_plot!

end # module
