module FastInterpolations

import LinearAlgebra
using LinearAlgebra: Tridiagonal, lu, ldiv!
using Preferences: @load_preference, @set_preferences!
using AdaptiveArrayPools

# Operation types (must be first - used by all interp files)
include("ops.jl")

# Boundary condition types
include("bc_types.jl")

# Shared internal utilities
include("utils.jl")

# Kernel functions (pure math, no dependencies)
include("linear_kernels.jl")
include("cubic_kernels.jl")
include("constant_kernels.jl")

# Linear interpolation
include("linear_interp.jl")

# Constant (step) interpolation
include("constant_interp.jl")

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
export constant_interp, constant_interp!, ConstantInterpolant
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!

# Boundary condition types
export AbstractBC, PointBC, Deriv1, Deriv2, BCPair
export NaturalBC, ClampedBC, PeriodicBC

# Derivative view functions for interpolants
export deriv1, deriv2

end # module
