# ========================================
# PHS — Include Aggregator
# ========================================
#
# Load order:
#   kernels   — pure math, no deps
#   stencil   — Φ matrix construction + unique-stencil precomputation
#   types     — PHSInterpolantND + PHSLogTransform struct definitions
#   eval      — evaluation engine (base-node lookup, stencil eval, blending)
#   interp    — constructor (phs_interp) + callable overloads
#   oneshot   — one-shot public API (phs_interp with 3-arg + phs_interp!)

include("phs_kernels.jl")
include("phs_stencil.jl")
include("phs_types.jl")
include("phs_eval.jl")
include("phs_interpolant.jl")
include("phs_oneshot.jl")

# ======================================================
# Precompile workload — runs only during `Pkg.precompile`
# (cached in the .ji file, so first-call JIT overhead is
# eliminated at runtime for the most common Float64/3D pattern)
# ======================================================
using PrecompileTools: @compile_workload

@compile_workload begin
    _x  = range(0.0, 1.0, 5)
    _gr = (_x, _x, _x)
    _da = [exp(-0.5 * ((i - 3.0)^2 + (j - 3.0)^2 + (k - 3.0)^2))
           for i in 1:5, j in 1:5, k in 1:5]

    # No-transform, K=3 (degree=3) — most common heavy-weight case
    _it3 = phs_interp(_gr, _da; stencil_size = 3, degree = 3, blend_factor = 1.5)
    _out = zeros(3)
    _qs  = ([0.3, 0.5, 0.7], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    _it3(_out, _qs)
    _it3(_out, _qs; deriv = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}()))
    _it3(_out, _qs; deriv = (DerivOp{2}(), DerivOp{0}(), DerivOp{0}()))
    _it3((0.5, 0.5, 0.5))
    _it3((0.5, 0.5, 0.5); deriv = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}()))
    _it3((0.5, 0.5, 0.5); deriv = (DerivOp{2}(), DerivOp{0}(), DerivOp{0}()))

    # Log-transform, K=3 — compile _phs_eval_with_transform paths
    _it_t = phs_interp(_gr, _da; stencil_size = 3, degree = 3, blend_factor = 1.5,
                       reference_interp = ConstantRef(1.0))
    _it_t(_out, _qs)
    _it_t(_out, _qs; deriv = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}()))
    _it_t(_out, _qs; deriv = (DerivOp{2}(), DerivOp{0}(), DerivOp{0}()))
    _it_t((0.5, 0.5, 0.5))
    _it_t((0.5, 0.5, 0.5); deriv = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}()))
    _it_t((0.5, 0.5, 0.5); deriv = (DerivOp{2}(), DerivOp{0}(), DerivOp{0}()))
end
