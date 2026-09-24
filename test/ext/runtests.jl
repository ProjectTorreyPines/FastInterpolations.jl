# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    EXTENSION TESTS (AD / Recipes / Symbolics)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Entrypoint for extension tests, run in a SEPARATE CI job (clean Julia process)
# to prevent ChainRulesCore contamination from the core test suite.
#
# Order matters: Enzyme MUST load before ChainRulesCore.
# ForwardDiff/Zygote transitively load CRC, so Enzyme tests must finish first.
#
# Usage:
#   CI:    Pkg.test(test_args=["ext/runtests.jl"])
#   Local: cc-julia-test-runner . ext/runtests.jl
#
# EXT_TEST_CHUNK splits the suite across CI jobs (each chunk = its own process,
# which dissolves the Enzyme-before-CRC ordering constraint between chunks):
#   enzyme → only the Enzyme tests (the dominant cost: ~36 min on Julia 1.12)
#   rest   → everything else (never loads Enzyme)
#   unset  → full suite in the documented order (local runs and CI.yml unchanged)

using Test
using FastInterpolations

const EXT_CHUNK = get(ENV, "EXT_TEST_CHUNK", "all")
EXT_CHUNK in ("all", "enzyme", "rest") ||
    error("EXT_TEST_CHUNK=$(repr(EXT_CHUNK)) — expected \"enzyme\", \"rest\", or unset (full suite)")

# Shared constants (mirrored from top-level runtests.jl)
const ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240

# ── Extension tests (order matters!) ──────────────────────────────────
if EXT_CHUNK in ("all", "enzyme")
    include("test_autodiff_Enzyme.jl")
end
if EXT_CHUNK in ("all", "rest")
    include("test_autodiff_ForwardDiff.jl")
    include("test_linear_dual_grid.jl")   # Linear 1D grid-side Dual
    include("test_hermite_dual_grid.jl")  # Hermite family grid-side Dual
    include("test_constant_quadratic_dual_grid.jl")  # Constant+Quadratic grid-side Dual
    include("test_cubic_dual_grid.jl")              # Cubic grid-side Dual
    include("test_series_dual_grid.jl")             # All methods × Series × Dual grid
    include("test_nd_dual_grid.jl")                 # ND (Cubic/Quadratic/Hetero) Dual grid
    include("test_autodiff_Zygote.jl")
    include("test_autodiff_hetero.jl")
    include("test_hermite_rrule.jl")
    include("test_symbolics.jl")
    include("test_recipes.jl")
end
