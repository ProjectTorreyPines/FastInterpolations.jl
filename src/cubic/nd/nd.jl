# ========================================
# ND Cubic Interpolation Module Aggregator
# ========================================
#
# N-dimensional cubic interpolation with precomputed coefficients.
#
# File Organization:
# - Generic ND files: nd_types.jl, nd_utils.jl, nd_math.jl, nd_build.jl, nd_eval.jl
# - 2D specialized files: nd_*_2d.jl (temporary, will be deprecated)
#
# Include order is critical for dependency resolution:
#   1. Types first (no code dependencies)
#   2. Utils (depends on core types)
#   3. Math/kernels (pure functions)
#   4. Build (depends on types + math)
#   5. Eval (depends on types + math)
#   6. API (routes to all backends)

# ========================================
# 1. Core Types (generic ND)
# ========================================
include("nd_types.jl")           # AbstractCoeffStrategy, CubicInterpolantND, NodalDerivativesND

# ========================================
# 2. Utilities
# ========================================
include("nd_utils.jl")           # Per-axis resolution helpers

# ========================================
# 3. Mathematical Functions
# ========================================
include("nd_math.jl")            # Hermite basis functions, 1D moment→deriv conversion

# ========================================
# 4. 2D Specialized Implementation (temporary)
# ========================================
# These files implement optimized 2D path and will be deprecated
# once CubicInterpolantND is validated for performance parity.

include("nd_types_2d.jl")        # CubicInterpolant2D, NodalDerivatives2D (aliases: BicubicInterpolant)
include("nd_math_2d.jl")         # 2D batch solvers (SIMD optimized)
include("nd_build_2d.jl")        # 2D coefficient computation
include("nd_eval_2d.jl")         # 2D evaluation

# ========================================
# 5. Generic ND Implementation
# ========================================
include("nd_build.jl")           # Generic ND coefficient computation
include("nd_eval.jl")            # Generic ND evaluation (@generated tensor product)

# ========================================
# 6. Public API
# ========================================
include("nd_api.jl")             # cubic_interp() for 2D (and future ND)
