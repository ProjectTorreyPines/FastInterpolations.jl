# Core module aggregator - shared foundations
# Order: types → ops → bc → grid → search → utils → periodic → nd_utils → series

include("abstract_types.jl")   # 1. AbstractInterpolant, AbstractSeriesInterpolant
include("eval_ops.jl")         # 2. AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2
include("bc_types.jl")         # 3. Boundary condition types
include("polyfit_kernels.jl")       # 4. Boundary condition computation kernels (Lagrange, etc.)
include("grid_spacing.jl")     # 5. ScalarSpacing, VectorSpacing
include("search.jl")           # 6. Search policy + interval search
include("factory.jl")          # 6b. User-facing factory functions (Search, Extrap, Side)
include("utils.jl")            # 7. Shared utilities (1D)
include("periodic.jl")         # 8. Periodic BC helpers (wrapping, validation, exclusive endpoint)
include("nd_utils.jl")         # 9. ND-specific utilities (shared by constant/linear/cubic ND)
include("thomas_lu_solver.jl") # 10. Thomas algorithm (TDMA) solvers for tridiagonal systems
include("series_utils.jl")     # 11. Series validation helpers
include("series_matrix.jl")    # 12. Lazy transpose infrastructure
include("series_wrapper.jl")   # 13. Series input wrapper + NamedSeriesInterpolant
