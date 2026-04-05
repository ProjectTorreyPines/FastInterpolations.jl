# PCHIP interpolation module aggregator
# Order: types → slopes → oneshot → interpolant

include("pchip_types.jl")        # 1. PchipInterpolant1D struct
include("pchip_slopes.jl")       # 2. _pchip_slopes! (Fritsch-Carlson algorithm)
include("pchip_oneshot.jl")      # 3. pchip_interp / pchip_interp! (scalar, vector, in-place)
include("pchip_interpolant.jl")  # 4. 2-arg pchip_interp + protocol traits
include("pchip_adjoint.jl")     # 5. PchipAdjoint1D (slope adjoint + Hermite scatter)
# Integration: dispatched via AbstractLocalCubicInterpolant1D (hermite_integrate.jl)
