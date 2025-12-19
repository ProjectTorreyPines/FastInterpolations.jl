module FastInterpolations

import LinearAlgebra
using LinearAlgebra: Tridiagonal, lu, ldiv!

# Shared internal utilities
include("utils.jl")

# Linear interpolation
include("linear_interp.jl")

# Cubic spline interpolation
include("cubic_interp.jl")
include("cubic_interp_autocache.jl")

# Exports
export linear_interp, linear_interp!, LinearInterpolant
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!, cubic_cache_stats
export get_cubic_cache

end # module
