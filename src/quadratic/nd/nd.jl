# ========================================
# ND Quadratic Interpolation Module Aggregator
# ========================================
#
# N-dimensional quadratic interpolation with precomputed coefficients.
#
# Include order is critical for dependency resolution:
#   1. Types first (no code dependencies)
#   2. Build (depends on types + 1D solver)
#   3. Eval (depends on types + 1D kernels)
#   4. API (public interface)
#
# Note: Shared ND utilities (NodalDerivativesND, _resolve_*_nd, etc.)
# are in src/core/nd_utils.jl.
# 1D quadratic kernels are in src/quadratic/quadratic_kernels.jl.
# 1D quadratic solver (recurrence) is in src/quadratic/quadratic_solver.jl.

# ========================================
# 1. Core Types
# ========================================
include("quadratic_nd_types.jl")     # QuadraticInterpolantND struct + accessors

# ========================================
# 2. Coefficient Construction
# ========================================
include("quadratic_nd_build.jl")     # ND partial derivative computation via quadratic recurrence

# ========================================
# 3. Evaluation
# ========================================
include("quadratic_nd_eval.jl")      # Callable interface + @generated tensor product kernel

# ========================================
# 4. Public API
# ========================================
include("quadratic_nd_interpolant.jl")  # quadratic_interp() constructor + BC resolution + builder
include("quadratic_nd_oneshot.jl")      # quadratic_interp() one-shot + pool-based backends

# ========================================
# 5. Adjoint (Transpose) Operator
# ========================================
include("quadratic_nd_adjoint_types.jl")  # QuadraticAdjointND struct + protocol accessors
include("quadratic_nd_adjoint.jl")        # Scatter, build adjoint, apply, constructor
