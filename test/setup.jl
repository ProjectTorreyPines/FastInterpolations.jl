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
    # LTS keeps a small 240-byte margin for genuine warm-path noise; 1.12+ is strict (0).
    const ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240
    const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240
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
