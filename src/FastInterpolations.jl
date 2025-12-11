module FastInterpolations

using LinearAlgebra: Tridiagonal, lu, ldiv!

# Linear interpolation
include("linear_interp.jl")

# Cubic spline interpolation
include("cubic_interp.jl")
include("cubic_interp_autocache.jl")

# Exports
export linear_interp, linear_interp!, LinearInterpCallable
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpCallable
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!, cubic_cache_stats

end # module
