# Hetero ND — aggregator
# ND interpolation with per-axis method specification.
# Supports OnTheFly (sequential 1D) and PreCompute (precomputed partials) strategies.
# Order: types → build → constructor → eval (on-the-fly) → eval (precomputed)

include("hetero_types.jl")
include("hetero_build.jl")
include("hetero_interpolant.jl")
include("hetero_eval.jl")
include("hetero_precomputed_eval.jl")
include("hetero_oneshot.jl")
include("hetero_nointerp.jl")
