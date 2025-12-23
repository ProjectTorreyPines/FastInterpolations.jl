module FastInterpolations

import LinearAlgebra
using LinearAlgebra: Tridiagonal, lu, ldiv!

# Boundary condition types (must be first - used by utils.jl)
include("bc_types.jl")

# Shared internal utilities
include("utils.jl")

# Linear interpolation
include("linear_interp.jl")

# Cubic spline interpolation
include("cubic_types.jl")      # Type definitions (PeriodicData, CubicSplineCache, CubicInterpolant)
include("cubic_solver.jl")     # Cache builders and system solvers
include("cubic_eval.jl")       # Evaluation functions
include("cubic_interp.jl")     # Public API
include("cubic_autocache.jl")

# Exports
export linear_interp, linear_interp!, LinearInterpolant
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!, cubic_cache_stats
export get_cubic_cache

# Boundary condition types
export AbstractBC, PointBC, D1, D2, BCPair, PeriodicBC

end # module
