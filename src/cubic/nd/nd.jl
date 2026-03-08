# ========================================
# ND Cubic Interpolation Module Aggregator
# ========================================
#
# N-dimensional cubic interpolation with precomputed coefficients.
#
# Include order is critical for dependency resolution:
#   1. Types first (no code dependencies)
#   2. Math/kernels (pure functions)
#   3. Build (depends on types + math)
#   4. Eval (depends on types + math)
#   5. API (public interface)
#
# Note: Shared ND utilities (_resolve_*_nd, _validate_nd_grids, etc.)
# are now in src/core/nd_utils.jl for reuse by constant/linear ND.

# ========================================
# 1. Core Types
# ========================================
include("cubic_nd_types.jl")     # AbstractCoeffStrategy, CubicInterpolantND, NodalDerivativesND

# ========================================
# 3. Mathematical Functions
# ========================================
include("cubic_nd_math.jl")      # Hermite basis functions, 1D moment→deriv conversion

# ========================================
# 4. Generic ND Implementation
# ========================================
include("cubic_nd_build.jl")     # Generic ND coefficient computation
include("cubic_nd_eval.jl")      # Generic ND evaluation (@generated tensor product)

# ========================================
# 5. Public API
# ========================================
include("cubic_nd_interpolant.jl")  # cubic_interp() constructor + internal builders
include("cubic_nd_oneshot.jl")      # cubic_interp() one-shot + pool-based backends

# ========================================
# 6. ND Adjoint Operator
# ========================================
include("cubic_nd_adjoint_types.jl")  # _NDAdjointAnchor, CubicAdjointND
include("cubic_nd_adjoint.jl")        # cubic_adjoint() ND constructor + apply pipeline
