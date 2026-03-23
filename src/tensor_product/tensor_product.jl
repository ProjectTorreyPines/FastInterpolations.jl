# Tensor Product ND — aggregator
# ND interpolation with per-axis method specification.
# Supports OnTheFly (sequential 1D) and PreCompute (precomputed partials) strategies.
# Order: types → build → constructor → eval (on-the-fly) → eval (precomputed)

include("tensor_product_types.jl")
include("tensor_product_build.jl")
include("tensor_product_interpolant.jl")
include("tensor_product_eval.jl")
include("tensor_product_precomputed_eval.jl")
include("tensor_product_oneshot.jl")
include("tensor_product_nointerp.jl")
