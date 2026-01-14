# Linear interpolation module aggregator
# Order: kernels → interp → anchor → multi

include("linear_kernels.jl")       # 1. Pure math kernels
include("linear_interp.jl")        # 2. LinearInterpolant + API
include("linear_anchor.jl")        # 3. Anchored queries
include("multi_linear_interp.jl")  # 4. Multi-Y interpolation
