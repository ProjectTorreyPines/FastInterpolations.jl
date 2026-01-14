# Constant interpolation module aggregator
# Order: kernels → interp → anchor → multi

include("../constant_kernels.jl")       # 1. Pure math kernels
include("../constant_interp.jl")        # 2. ConstantInterpolant + API
include("../constant_anchor.jl")        # 3. Anchored queries
include("../multi_constant_interp.jl")  # 4. Multi-Y interpolation
