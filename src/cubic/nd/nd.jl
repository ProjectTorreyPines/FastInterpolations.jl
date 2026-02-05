# ========================================
# ND Cubic Interpolation Module Aggregator
# ========================================
#
# N-dimensional cubic interpolation with precomputed coefficients.
#
# Include order is critical for dependency resolution:
#   1. Types first (no code dependencies)
#   2. Utils (depends on core types)
#   3. Math/kernels (pure functions)
#   4. Build (depends on types + math)
#   5. Eval (depends on types + math)
#   6. API (public interface)

# ========================================
# 1. Core Types
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
# 4. Generic ND Implementation
# ========================================
include("nd_build.jl")           # Generic ND coefficient computation
include("nd_eval.jl")            # Generic ND evaluation (@generated tensor product)

# ========================================
# 5. Public API
# ========================================
include("nd_api.jl")             # cubic_interp() for ND
