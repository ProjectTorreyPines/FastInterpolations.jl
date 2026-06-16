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
    const ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240
    const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240
end

# Helper for @test_broken throw-pins (test_phs_broken_pins.jl). Returns `true`
# iff calling `f()` raises an exception of type `T` (or a subtype). Pairs with
# @test_broken to pin "this SHOULD throw once the bug is fixed": while the code
# still silently misbehaves, `is_throwing` returns `false` → recorded as Broken;
# the moment a follow-up PR adds the validation, it returns `true` → @test_broken
# reports an Unexpected Pass, signalling "promote me to @test".
@testsnippet PHSBrokenHelpers begin
    is_throwing(f, T::Type) = try
        f()
        false
    catch e
        e isa T
    end
end

# DuckFloat5 type + shared 1D/2D fixtures for the duck-typing comprehensive
# tests (test_duck_typing_comprehensive.jl). Extracted as a snippet so the
# testitem split inside that file can reuse the same setup without copy-paste.
@testsnippet DuckTypeSetup begin
    struct MyDuck
        v::Float64
    end

    Base.:+(a::MyDuck, b::MyDuck) = MyDuck(a.v + b.v)
    Base.:-(a::MyDuck, b::MyDuck) = MyDuck(a.v - b.v)
    Base.:*(a::Real, b::MyDuck) = MyDuck(a * b.v)
    Base.:*(a::MyDuck, b::Real) = MyDuck(a.v * b)

    _val(d::MyDuck) = d.v

    # 1D grids
    x_vec = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_rng = range(0.0, 6.0, 7)
    xq = 2.7
    xq_vec = [1.5, 2.7, 4.3]

    # Polynomial data for exact-result testing
    y_linear = MyDuck.(2 .* collect(x_vec) .+ 1)
    y_quad = MyDuck.(collect(x_vec) .^ 2 .- collect(x_vec) .+ 1)
    y_cubic = MyDuck.(collect(x_vec) .^ 3 ./ 6 .- collect(x_vec) ./ 2 .+ 1)
    y_generic = MyDuck.([1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.5])

    # Flat (Float64) reference copies for correctness checks
    y_linear_flat = _val.(y_linear)
    y_quad_flat = _val.(y_quad)
    y_cubic_flat = _val.(y_cubic)
    y_generic_flat = _val.(y_generic)

    # 2D grids
    xg = [0.0, 1.0, 2.0, 3.0]
    yg = [0.0, 1.0, 2.0, 3.0]
    xg_r = range(0.0, 3.0, 4)
    yg_r = range(0.0, 3.0, 4)
    data_2d = [MyDuck(xi + 2yj) for xi in xg, yj in yg]
    data_2d_flat = _val.(data_2d)
    q2d = (1.5, 1.5)
end
