# Quadratic interpolation module aggregator
# Order: kernels → solver → interp → anchor → multi

include("../quadratic_kernels.jl")       # 1. Pure math kernels
include("../quadratic_solver.jl")        # 2. Coefficient computation
include("../quadratic_interp.jl")        # 3. QuadraticInterpolant + API
include("../quadratic_anchor.jl")        # 4. Anchored queries
include("../multi_quadratic_interp.jl")  # 5. Multi-Y interpolation
