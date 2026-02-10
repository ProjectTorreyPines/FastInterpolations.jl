# Integration module aggregator
# Order: ops → common_1d → common_nd → kernels → api → fulldomain

include("integrate_ops.jl")        # 1. Operation tag types
include("integrate_common_1d.jl")  # 2. 1D cell-wise integration helpers
include("integrate_common_nd.jl")  # 3. ND integration helpers
include("integrate_kernels.jl")    # 4. Per-method integration kernels
include("integrate_api.jl")        # 5. Public `integrate` dispatch
include("integrate_fulldomain.jl") # 6. Full-domain & cumulative integration
