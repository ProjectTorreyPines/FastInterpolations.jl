# Linear interpolation module aggregator
# Order: kernels → types → oneshot → interpolant → anchor → series → ND

include("linear_kernels.jl")       # 1. Pure math kernels
include("linear_types.jl")         # 2. LinearInterpolant struct
include("linear_oneshot.jl")       # 3. 3-arg API (linear_interp!, linear_interp)
include("linear_interpolant.jl")   # 4. 2-arg API, callable
include("linear_anchor.jl")        # 5. Anchored queries
include("linear_adjoint.jl")       # 5b. Adjoint operator (Wᵀ)
include("linear_series_interp.jl") # 6. Multi-Y series interpolation (replaces multi_linear_interp.jl)
include("nd/nd.jl")                # 7. N-dimensional multilinear interpolation
