# Core module aggregator - shared foundations
# Order: types → ops → bc → grid → cached_range → search → utils → periodic → nd_utils → series

include("abstract_types.jl")   # 1. AbstractInterpolant, AbstractSeriesInterpolant
include("eval_ops.jl")         # 2. AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2
include("bc_types.jl")         # 3. Boundary condition types
include("interp_method_types.jl") # 3b. Per-axis method specification (CubicMethod, etc.)
include("coeff_types.jl")         # 3c. AbstractCoeffStrategy, PreCompute, OnTheFly, AutoCoeffs
include("polyfit_kernels.jl")       # 4. Boundary condition computation kernels (Lagrange, etc.)
include("axis_types.jl")       # 4b. Axis struct definitions (_CachedRange / _CachedVector / _ExclusivePeriodicAxis) — central types-first
include("cached_range.jl")     # 5. _CachedRange methods (_to_float, _get_h, _resolve_axis, _cache_axis*)
include("cached_vector.jl")    # 5c. _CachedVector methods (build ctor, _get_h, _resolve_axis, _cache_axis*)
include("search.jl")           # 6. Search policy + interval search
include("idx_stencil.jl")      # 6a. _IdxStencil{K} — wrap-aware per-axis index stencil
include("factory.jl")          # 6b. User-facing factory functions (Search, Extrap, Side)
include("utils.jl")            # 7. Shared utilities (1D)
include("store_policy.jl")     # 7b. StorePolicy (copy vs reference storage) + own/ref helpers
include("periodic.jl")         # 8. Periodic BC helpers (wrapping, validation, exclusive endpoint)
include("periodic_axis.jl")    # 8b. _ExclusivePeriodicAxis wrapper (axis-side representation transform for `:exclusive` BC on Vector grids)
include("periodic_data.jl")    # 8c. _ExclusivePeriodicData wrapper (data-side cyclic-indexing companion)
include("anchor_common.jl")    # 8c. Shared _AnchorLoc + _anchor_loc (all methods, 1D/ND)
include("nd_utils.jl")            # 9. ND-specific utilities (shared by constant/linear/cubic ND)
include("query_protocol.jl")           # 9b. Query protocol (query_length, extract, eltype, validate)
include("interpolant_protocol.jl")     # 9c. Interpolant callable interface (1D + ND)
include("adjoint_protocol.jl")         # 9d. Adjoint callable interface (1D + ND)
include("nd_adjoint_scatter.jl")       # 9e. Shared ND adjoint scatter (_NDAdjointAnchor, _scatter_nd!)
include("thomas_lu_solver.jl") # 10. Thomas algorithm (TDMA) solvers for tridiagonal systems
include("series_utils.jl")     # 11. Series validation helpers
include("series_matrix.jl")    # 12. Lazy transpose infrastructure
include("series_wrapper.jl")   # 13. Series input wrapper + NamedSeriesInterpolant
