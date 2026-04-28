# Hermite interpolation module aggregator
# Shared types first, then per-family files (cubic, quadratic in future)

include("hermite_types.jl")            # 1. CubicHermiteInterpolant1D struct
include("hermite_slope_methods.jl")    # 2. AbstractSlopeMethod, PchipSlopes{BC}, etc.
include("hermite_local_slopes.jl")     # 3. _local_slope per-index + bc-dispatched boundary helpers
include("hermite_periodic_slopes.jl")  # 4. _periodic_secant / _periodic_cell_width wrap-aware primitives
include("hermite_eval.jl")             # 5. _hermite_eval_at_point, _hermite_vector_loop!
include("hermite_oneshot.jl")          # 6. hermite_interp / hermite_interp!
include("hermite_interpolant.jl")      # 7. 2-arg hermite_interp + protocol traits
include("hermite_adjoint.jl")          # 8. HermiteAdjoint1D + shared scatter core
include("hermite_integrate.jl")        # 9. integrate(itp::AbstractHermiteInterpolant1D)
