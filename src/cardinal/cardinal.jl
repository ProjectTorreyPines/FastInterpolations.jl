# Cardinal spline interpolation module aggregator
# Order: types → slopes → oneshot → interpolant

include("cardinal_types.jl")        # 1. CardinalInterpolant1D struct
include("cardinal_slopes.jl")       # 2. _cardinal_slopes!
include("cardinal_oneshot.jl")      # 3. cardinal_interp / cardinal_interp!
include("cardinal_interpolant.jl")  # 4. 2-arg cardinal_interp + protocol traits
include("cardinal_adjoint.jl")     # 5. CardinalAdjoint1D (adjoint operator)
# Integration: dispatched via AbstractLocalCubicInterpolant1D (hermite_integrate.jl)
