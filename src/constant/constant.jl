# Constant interpolation module aggregator
# Order: kernels → types → oneshot → interpolant → anchor → multi

include("constant_kernels.jl")       # 1. Pure math kernels
include("constant_types.jl")         # 2. ConstantInterpolant struct
include("constant_oneshot.jl")       # 3. 3-arg API (constant_interp!, constant_interp)
include("constant_interpolant.jl")   # 4. 2-arg API, callable
include("constant_anchor.jl")        # 5. Anchored queries
include("multi_constant_interp.jl")  # 6. Multi-Y interpolation
