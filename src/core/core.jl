# Core module aggregator - shared foundations
# Order: types → ops → bc → grid → search → utils → series

include("abstract_types.jl")   # 1. AbstractInterpolant, AbstractSeriesInterpolant
include("eval_ops.jl")         # 2. AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2
include("bc_types.jl")         # 3. Boundary condition types
include("polyfit_kernels.jl")       # 4. Boundary condition computation kernels (Lagrange, etc.)
include("grid_spacing.jl")     # 5. ScalarSpacing, VectorSpacing
include("search.jl")           # 6. Search policy + interval search
include("utils.jl")            # 7. Shared utilities
include("thomas_lu_solver.jl") # 8. Thomas algorithm (TDMA) solvers for tridiagonal systems
include("series_utils.jl")     # 9. Series validation helpers
include("series_matrix.jl")    # 10. Lazy transpose infrastructure
