# Cubic interpolation module aggregator
# Order: types → kernels → solver → eval → autocache → interp → anchor → interpolant → multi

include("../cubic_types.jl")        # 1. Type definitions
include("../cubic_kernels.jl")      # 2. Pure math kernels
include("../cubic_solver.jl")       # 3. Cache builders, system solvers
include("../cubic_eval.jl")         # 4. Evaluation functions
include("../cubic_autocache.jl")    # 5. Ring buffer cache
include("../cubic_interp.jl")       # 6. 4-arg API
include("../cubic_anchor.jl")       # 7. Anchored queries
include("../cubic_interpolant.jl")  # 8. 2-arg API, callable
include("../multi_cubic_interp.jl") # 9. Multi-Y interpolation
