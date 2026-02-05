# ========================================
# ND Constant Interpolation Module Aggregator
# ========================================
#
# N-dimensional constant (step) interpolation.
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
include("nd_types.jl")           # ConstantInterpolantND

# ========================================
# 2. Evaluation
# ========================================
include("nd_eval.jl")            # Scalar and batch evaluation

# ========================================
# 3. Public API
# ========================================
include("nd_api.jl")             # constant_interp() for ND
