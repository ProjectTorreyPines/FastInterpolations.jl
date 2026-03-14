# Constant interpolation module aggregator
# Order: kernels → types → oneshot → interpolant → anchor → series → nd

include("constant_kernels.jl")       # 1. Pure math kernels
include("constant_types.jl")         # 2. ConstantInterpolant struct
include("constant_oneshot.jl")       # 3. 3-arg API (constant_interp!, constant_interp)
include("constant_interpolant.jl")   # 4. 2-arg API, callable
include("constant_anchor.jl")        # 5. Anchored queries
include("constant_adjoint.jl")       # 5b. Adjoint operator (Wᵀ)
include("constant_series_interp.jl") # 6. Multi-Y series interpolation (replaces multi_constant_interp.jl)
include("nd/nd.jl")                  # 7. N-dimensional constant interpolation
