# Shared @testsnippets for FastInterpolations test suite.
#
# Loaded automatically by TestItemRunner via test/runtests.jl. Each
# @testsnippet here can be referenced by any @testitem via setup=[Name].

using TestItemRunner

# Allocation thresholds — used by ~63 test files.
# `using FastInterpolations` is auto-injected into every testitem, so
# accessing FastInterpolations.AdaptiveArrayPools.RUNTIME_CHECK is safe.
@testsnippet AllocConstants begin
    const AAP_RUNTIME_CHECK = FastInterpolations.AdaptiveArrayPools.RUNTIME_CHECK
    const _COV_OVERHEAD = 16
    const ALLOC_THRESHOLD = (VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240) + _COV_OVERHEAD
    const ND_ALLOC_THRESHOLD = (VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240) + _COV_OVERHEAD
end
