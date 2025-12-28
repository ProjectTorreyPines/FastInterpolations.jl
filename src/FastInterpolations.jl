module FastInterpolations

import LinearAlgebra
using LinearAlgebra: Tridiagonal, lu, ldiv!

# Operation types (must be first - used by all interp files)
include("ops.jl")

# Boundary condition types
include("bc_types.jl")

# Shared internal utilities
include("utils.jl")

# Kernel functions (pure math, no dependencies)
include("linear_kernels.jl")
include("cubic_kernels.jl")

# Linear interpolation
include("linear_interp.jl")

# Cubic spline interpolation
include("cubic_types.jl")       # Type definitions (PeriodicData, CubicSplineCache, CubicInterpolant)
include("cubic_solver.jl")      # Cache builders and system solvers
include("cubic_eval.jl")        # Evaluation functions
include("cubic_autocache.jl")   # Ring buffer cache for reusing LU factorizations
include("cubic_interp.jl")      # 4-arg API, helper functions
include("cubic_interpolant.jl") # 2-arg API, CubicInterpolant callable

# Derivative view wrapper (depends on both CubicInterpolant and LinearInterpolant)
include("derivative_view.jl")

# Exports
export linear_interp, linear_interp!, LinearInterpolant
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!, cubic_cache_stats

# Boundary condition types
export AbstractBC, PointBC, Deriv1, Deriv2, BCPair
export NaturalBC, ClampedBC, PeriodicBC

# Evaluation operation types (for advanced use)
export AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2

# Derivative functions for interpolants
export derivative, derivative2

end # module
