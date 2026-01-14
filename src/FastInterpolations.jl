module FastInterpolations

import LinearAlgebra
using LinearAlgebra: Tridiagonal, lu, ldiv!
using Preferences: @load_preference, @set_preferences!
using AdaptiveArrayPools

# Module aggregators (encapsulate include order complexity)
include("core/core.jl")
include("linear/linear.jl")
include("constant/constant.jl")
include("quadratic/quadratic.jl")
include("cubic/cubic.jl")

# Derivative view wrapper (depends on all interpolant types)
include("derivative_view.jl")

# Exports
export AbstractInterpolant, AbstractMultiInterpolant
export linear_interp, linear_interp!, LinearInterpolant, LinearMultiInterpolant
export constant_interp, constant_interp!, ConstantInterpolant, ConstantMultiInterpolant
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant
export CubicMultiInterpolant, MultiCubicInterpolant  # MultiCubicInterpolant is alias for backward compat
export precompute_transpose!  # Pre-allocate point-contiguous layout for scalar queries
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!
export quadratic_interp, quadratic_interp!, QuadraticInterpolant, QuadraticMultiInterpolant

# Boundary condition types
export AbstractBC, PointBC, Deriv1, Deriv2, BCPair
export NaturalBC, ClampedBC, PeriodicBC, MinCurvFit, ParabolaFit
export Left, Right

# Derivative view functions for interpolants
export deriv1, deriv2

# Operation types (for derivative dispatch)
export AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2

end # module
