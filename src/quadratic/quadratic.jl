# Quadratic interpolation module aggregator
# Order: kernels → solver → types → oneshot → interpolant → anchor → multi

include("quadratic_kernels.jl")       # 1. Pure math kernels
include("quadratic_solver.jl")        # 2. Coefficient computation (QuadraticBC, _compute_quadratic_coeffs)
include("quadratic_types.jl")         # 3. QuadraticInterpolant struct
include("quadratic_oneshot.jl")       # 4. 3-arg API (quadratic_interp!, quadratic_interp)
include("quadratic_interpolant.jl")   # 5. 2-arg API, callable
include("quadratic_anchor.jl")        # 6. Anchored queries
include("quadratic_series_interp.jl") # 7. Multi-Y series interpolation (replaces multi_quadratic_interp.jl)
