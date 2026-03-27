# Cubic interpolation module aggregator
# Order: types → kernels → solver → eval → cache_pool → cache → oneshot → anchor → interpolant → series → nd

include("cubic_types.jl")        # 1. Type definitions
include("cubic_kernels.jl")      # 2. Pure math kernels
include("cubic_solver.jl")       # 3. Cache builders, system solvers
include("cubic_eval.jl")         # 4. Evaluation functions
include("cubic_cache_pool.jl")   # 5. Ring buffer cache pool
include("cubic_cache.jl")        # 6. CubicSplineCache constructor
include("cubic_oneshot.jl")      # 7. 4-arg API (cubic_interp!, cubic_interp)
include("cubic_anchor.jl")       # 8. Anchored queries
include("cubic_oneshot_series.jl") # 8a. One-shot Series API (Strategy B)
include("cubic_interpolant.jl")  # 9. 2-arg API, callable
include("cubic_adjoint.jl")      # 9b. Adjoint operator (W^T)
include("cubic_series_interp.jl") # 10. Series interpolation (CubicSeriesInterpolant)
include("nd/nd.jl")              # 11. N-dimensional cubic (CubicInterpolantND)
