# Core module aggregator - shared foundations
# Order: types → ops → bc → grid → search → utils → series

include("abstract_types.jl")   # 1. AbstractInterpolant, AbstractSeriesInterpolant
include("eval_ops.jl")         # 2. AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2
include("bc_types.jl")         # 3. Boundary conditions
include("grid_spacing.jl")     # 4. ScalarSpacing, VectorSpacing
include("search.jl")           # 5. Search policy + interval search
include("utils.jl")            # 6. Shared utilities (uses _find_interval alias)
include("series_utils.jl")     # 7. Series validation helpers
include("series_matrix.jl")    # 8. Lazy transpose infrastructure
