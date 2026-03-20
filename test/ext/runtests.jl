# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    EXTENSION TESTS (AD / Symbolics)                      ║
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

using Test
using FastInterpolations

# Shared constants (mirrored from top-level runtests.jl)
const ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240

# ── Extension tests (order matters!) ──────────────────────────────────
include("test_autodiff_Enzyme.jl")
include("test_autodiff_ForwardDiff.jl")
include("test_autodiff_Zygote.jl")
include("test_symbolics.jl")
