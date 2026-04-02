# Akima interpolation module aggregator
# Order: types → slopes → oneshot → interpolant

include("akima_types.jl")        # 1. AkimaInterpolant1D struct
include("akima_slopes.jl")       # 2. _akima_slopes! (Akima 1970 algorithm)
include("akima_oneshot.jl")      # 3. akima_interp / akima_interp!
include("akima_interpolant.jl")  # 4. 2-arg akima_interp + protocol traits
