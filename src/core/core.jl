# Core module aggregator - shared foundations
# Order: types → ops → bc → grid → utils → series

include("abstract_types.jl")   # 1. AbstractInterpolant, AbstractSeriesInterpolant
include("eval_ops.jl")         # 2. AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2
include("bc_types.jl")         # 3. Boundary conditions
include("grid_spacing.jl")     # 4. ScalarSpacing, VectorSpacing
include("utils.jl")            # 5. Shared utilities
include("series_utils.jl")     # 6. Series validation helpers
include("series_matrix.jl")    # 7. Lazy transpose infrastructure
