module FastInterpolations

import LinearAlgebra
using LinearAlgebra: Tridiagonal, lu, ldiv!
using Preferences: @load_preference, @set_preferences!
using AdaptiveArrayPools

# Abstract type hierarchy (must be first - defines AbstractInterpolant, AbstractMultiInterpolant)
include("abstract_types.jl")

# Operation types (used by all interp files)
include("ops.jl")

# Boundary condition types
include("bc_types.jl")

# Grid spacing types (ScalarSpacing for Range, VectorSpacing for Vector)
include("grid_spacing.jl")

# Shared internal utilities
include("utils.jl")

# Kernel functions (pure math, no dependencies)
include("linear_kernels.jl")
include("cubic_kernels.jl")
include("constant_kernels.jl")
include("quadratic_kernels.jl")

# Linear interpolation
include("linear_interp.jl")
include("linear_anchor.jl")    # Anchored query for ultra-fast evaluation (after LinearInterpolant)
include("multi_linear_interp.jl") # Multi-Y linear interpolation

# Constant (step) interpolation
include("constant_interp.jl")
include("constant_anchor.jl")  # Anchored query for ultra-fast evaluation (after ConstantInterpolant)
include("multi_constant_interp.jl") # Multi-Y constant interpolation

# Cubic spline interpolation
include("cubic_types.jl")       # Type definitions (PeriodicData, CubicSplineCache, CubicInterpolant)
include("cubic_solver.jl")      # Cache builders and system solvers
include("cubic_eval.jl")        # Evaluation functions
include("cubic_autocache.jl")   # Ring buffer cache for reusing LU factorizations
include("cubic_interp.jl")      # 4-arg API, helper functions
include("cubic_anchor.jl")      # Anchored query for ultra-fast evaluation
include("cubic_interpolant.jl") # 2-arg API, CubicInterpolant callable
include("multi_cubic_interp.jl") # Multi-Y cubic interpolation

# Quadratic spline interpolation
include("quadratic_solver.jl")       # Coefficient computation (secants, d[], a[])
include("quadratic_interp.jl")      # Public API (quadratic_interp, quadratic_interp!)
include("quadratic_anchor.jl")      # Anchored query for ultra-fast evaluation (after QuadraticInterpolant)
include("multi_quadratic_interp.jl") # Multi-Y quadratic interpolation

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
