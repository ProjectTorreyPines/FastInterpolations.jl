# Hermite interpolation module aggregator
# Shared types first, then per-family files (cubic, quadratic in future)

include("hermite_types.jl")          # 1. HermiteInterpolant1D struct
include("hermite_eval.jl")           # 2. _hermite_eval_at_point, _hermite_vector_loop!
include("hermite_oneshot.jl")        # 3. hermite_interp / hermite_interp!
include("hermite_interpolant.jl")    # 4. 2-arg hermite_interp + protocol traits
include("hermite_adjoint.jl")        # 5. HermiteAdjoint1D + shared scatter core
include("hermite_integrate.jl")      # 6. integrate(itp::AbstractHermiteInterpolant1D)
