# ND Cubic Hermite (user-supplied partials) module aggregator.

include("hermite_nd_types.jl")        # 1. HermitePartials, CubicHermiteInterpolantND
include("hermite_nd_partials.jl")     # 2. HermitePartials constructor + validation
include("hermite_nd_build.jl")        # 3. _pack_and_extend_nodal_derivs + BC/size validation
include("hermite_nd_interpolant.jl")  # 4. outer ctor + callable + _locate_cell / _eval_at_cell + hermite_interp(grids, data, partials)
include("hermite_nd_oneshot.jl")      # 5. hermite_interp / hermite_interp! one-shot variants
