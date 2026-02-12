# ========================================
# ND Linear Interpolation Module Aggregator
# ========================================
#
# N-dimensional multilinear (bilinear/trilinear/n-linear) interpolation.
#
# Include order:
#   1. Types (no dependencies except core)
#   2. Eval (depends on types)
#   3. API (depends on types + eval)
#
# Note: Shared ND utilities are in src/core/nd_utils.jl

# ========================================
# 1. Core Types
# ========================================
include("linear_nd_types.jl")    # LinearInterpolantND

# ========================================
# 2. Evaluation
# ========================================
include("linear_nd_eval.jl")     # Multilinear interpolation kernel

# ========================================
# 3. Public API
# ========================================
include("linear_nd_interpolant.jl")  # linear_interp() constructor + grid conversion
include("linear_nd_oneshot.jl")      # linear_interp() one-shot + zero-alloc backends
