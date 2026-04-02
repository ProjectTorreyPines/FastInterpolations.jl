# Hermite interpolation module aggregator
# Shared types first, then per-family files (cubic, quadratic in future)

include("hermite_types.jl")              # 1. Shared: AbstractDataWrapper, Hermite, CubicHermiteInterpolant1D
include("cubic_hermite_eval.jl")         # 2. Cubic: _cubic_hermite_eval_at_point, _cubic_hermite_vector_loop!
include("cubic_hermite_oneshot.jl")      # 3. Cubic: cubic_interp / cubic_interp! Hermite dispatch
include("cubic_hermite_interpolant.jl")  # 4. Cubic: 2-arg cubic_interp + protocol traits
include("cubic_hermite_adjoint.jl")      # 5. Adjoint: HermiteAdjoint1D + shared scatter core
include("cubic_hermite_integrate.jl")    # 6. Integration: integrate(itp, x0, x1)
