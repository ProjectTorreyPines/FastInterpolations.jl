# ========================================
# Duck-typed grid axis: contract tests
# ========================================
# Phase 0 of the ducktype-grid plan (claudedocs/plans/2026-07-21-ducktype-grid-plan.md):
#   1. Unordered grid eltype (Complex) → friendly ArgumentError at every public
#      build entry (guard funnel) — NOT a deep MethodError from search/sort internals.
#   2. Real-grid baselines pinned BEFORE any bound relaxation: values (cross-type
#      identity) AND `Base.promote_op` witness types (guards Phase 1 witness
#      homogenization against Real-type drift).
#   3. Dual-through-grid AD baseline (the `<:Real` faithfulness the relaxation
#      must preserve).

@testitem "Duck grid: unordered eltype → friendly ArgumentError (guard funnel)" begin
    xc = ComplexF64[0.0 + 0im, 1.0 + 0im, 2.0 + 0im, 3.0 + 0im, 4.0 + 0im]
    yc = [0.0, 1.0, 0.5, 2.0, 1.0]
    dyc = [1.0, 0.5, 0.0, -0.5, -1.0]

    # A friendly guard error: ArgumentError whose message names the ordering
    # requirement — never a MethodError from `isless` deep inside search/sort.
    is_guard_error(err) = err isa ArgumentError && occursin("isless", sprint(showerror, err))

    @testset "1D family factories" begin
        @test_throws ArgumentError linear_interp(xc, yc)
        @test_throws ArgumentError constant_interp(xc, yc)
        @test_throws ArgumentError quadratic_interp(xc, yc)
        @test_throws ArgumentError cubic_interp(xc, yc)
        @test_throws ArgumentError pchip_interp(xc, yc)
        @test_throws ArgumentError akima_interp(xc, yc)
        @test_throws ArgumentError cardinal_interp(xc, yc)
        @test_throws ArgumentError hermite_interp(xc, yc, dyc)
    end

    @testset "guard message names the requirement" begin
        err = try
            linear_interp(xc, yc)
            nothing
        catch e
            e
        end
        @test is_guard_error(err)
    end

    @testset "unified interp() entries" begin
        @test_throws ArgumentError interp(xc, yc; method = LinearInterp())
        # ND factory (2D) with one Complex axis (either position)
        xg3 = [0.0, 1.0, 2.0]
        xg5 = [0.0, 1.0, 2.0, 3.0, 4.0]
        data = [Float64(i + j) for i in 1:3, j in 1:5]
        @test_throws ArgumentError interp((xg3, xc), data; method = LinearInterp())
        @test_throws ArgumentError interp((xc[1:3], xg5), data; method = LinearInterp())
    end

    @testset "one-shot integrate entries" begin
        @test_throws ArgumentError integrate(xc, yc; method = LinearInterp())
        @test_throws ArgumentError cumulative_integrate(xc, yc; method = LinearInterp())
    end

    @testset "one-shot query entries (scalar / vector / in-place)" begin
        # Every allocating & in-place one-shot sibling must funnel through the
        # SAME guard as their persistent (x, y) factories — else a Complex grid
        # leaks a raw search-internal MethodError, not the friendly ArgumentError.
        # (Vector-alloc forms delegate to `interp!`, so guarding scalar + `!`
        # covers all three query shapes; asserting each shape pins that coverage.)
        q = 2.5 + 0im
        qs = ComplexF64[2.5 + 0im, 3.5 + 0im]
        out = zeros(ComplexF64, 2)
        @testset "scalar" begin
            @test_throws ArgumentError linear_interp(xc, yc, q)
            @test_throws ArgumentError constant_interp(xc, yc, q)
            @test_throws ArgumentError pchip_interp(xc, yc, q)
            @test_throws ArgumentError akima_interp(xc, yc, q)
            @test_throws ArgumentError cardinal_interp(xc, yc, q)
            @test_throws ArgumentError hermite_interp(xc, yc, dyc, q)
        end
        @testset "vector-alloc" begin
            @test_throws ArgumentError linear_interp(xc, yc, qs)
            @test_throws ArgumentError constant_interp(xc, yc, qs)
            @test_throws ArgumentError pchip_interp(xc, yc, qs)
            @test_throws ArgumentError akima_interp(xc, yc, qs)
            @test_throws ArgumentError cardinal_interp(xc, yc, qs)
            @test_throws ArgumentError hermite_interp(xc, yc, dyc, qs)
        end
        @testset "in-place" begin
            @test_throws ArgumentError linear_interp!(out, xc, yc, qs)
            @test_throws ArgumentError constant_interp!(out, xc, yc, qs)
            @test_throws ArgumentError pchip_interp!(out, xc, yc, qs)
            @test_throws ArgumentError akima_interp!(out, xc, yc, qs)
            @test_throws ArgumentError cardinal_interp!(out, xc, yc, qs)
            @test_throws ArgumentError hermite_interp!(out, xc, yc, dyc, qs)
        end
    end
end

@testitem "Duck grid: Real-grid baselines (cross-type value identity)" begin
    # Grid nodes exactly representable in every Real type under test; queries
    # likewise (1.5, 2.75). Pinned BEFORE the bound relaxation: these must stay
    # bit-identical through every phase.
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 1.0, 0.5, 2.0, 1.0]

    LIN_INT = 4.0                    # integrate(linear_interp(xf, y))
    CUB_INT = 4.555555555555555      # integrate(cubic_interp(xf, y))
    LIN_EVAL = 0.75                  # linear @ 1.5
    CUB_EVAL = 0.6145833333333334    # cubic @ 1.5

    @testset "Float64 reference pins" begin
        @test integrate(linear_interp(xf, y)) === LIN_INT
        @test integrate(cubic_interp(xf, y)) === CUB_INT
        @test linear_interp(xf, y)(1.5) === LIN_EVAL
        @test cubic_interp(xf, y)(1.5) === CUB_EVAL
    end

    @testset "grid eltype: $(T)" for T in (Int, Float32, Float64, Rational{Int}, BigFloat)
        xT = T <: Integer ? T.(0:4) : T.(xf)
        lin = linear_interp(xT, y)
        @test integrate(lin) == LIN_INT
        @test lin(1.5) == LIN_EVAL
        cub = cubic_interp(xT, y)
        # Thomas solve in BigFloat rounds differently in the last ulps — value
        # identity is pinned at rtol=1e-15 (house precedent), exact for the rest.
        if T === BigFloat
            @test isapprox(integrate(cub), CUB_INT; rtol = 1.0e-15)
            @test isapprox(cub(1.5), CUB_EVAL; rtol = 1.0e-15)
        else
            @test integrate(cub) == CUB_INT
            @test cub(1.5) == CUB_EVAL
        end
    end
end

@testitem "Duck grid: witness promote_op type pins (Real)" begin
    const FI = FastInterpolations

    # Pinned current behavior of the promotion witnesses on Real types. Phase 1
    # homogenizes these witnesses for units — these pins prove Real types do not
    # drift (bit-identical `Tout` contract).
    @testset "uniform $(T)" for (T, R) in (
            (Int, Float64),               # inv(Int)::Float64 floats the grid
            (Float32, Float32),
            (Float64, Float64),
            (BigFloat, BigFloat),
            (Rational{Int}, Rational{Int}),
        )
        @test Base.promote_op(FI._interp_op, T, T, T) === R
        @test Base.promote_op(FI._coeff_op, T, T) === R
        # order-2 witness (cubic z / quadratic a): on Real grids it must stay
        # identical to order-1 — the Unitful branch splits them (Y/X vs Y/X²),
        # but that split must not leak a widened Real `Tout`.
        @test Base.promote_op(FI._coeff_op2, T, T) === R
        @test Base.promote_op(FI._integrate_op, T, T, T) === R
        @test Base.promote_op(FI._inv_op, T) === R
    end

    @testset "mixed combos" begin
        @test Base.promote_op(FI._integrate_op, Int, Float64, Int) === Float64
        @test Base.promote_op(FI._interp_op, Float32, Float64, Float32) === Float64
        # inv(Int)::Float64 dominates Float32 values — current (pinned) behavior.
        @test Base.promote_op(FI._coeff_op, Int, Float32) === Float64
        @test Base.promote_op(FI._coeff_op2, Int, Float32) === Float64
    end
end

@testitem "Duck grid: Dual-through-grid AD baseline" begin
    using ForwardDiff

    # ∫ over a grid scaled by `a` scales linearly in `a` for linear interpolation
    # of fixed nodal values: d/da ∫(a·x) = ∫(x). Guards `<:Real` faithfulness
    # (Dual grids must keep working through every phase).
    xb = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 1.0, 0.5, 2.0, 1.0]
    base = integrate(linear_interp(xb, y))

    d = ForwardDiff.derivative(a -> integrate(linear_interp(a .* xb, y)), 1.0)
    @test d ≈ base rtol = 1.0e-14

    # Dual grid + Dual query through eval
    g = ForwardDiff.derivative(a -> linear_interp(a .* xb, y)(1.5 * a), 1.0)
    ref = let ε = 1.0e-7
        (linear_interp((1 + ε) .* xb, y)(1.5 * (1 + ε)) - linear_interp((1 - ε) .* xb, y)(1.5 * (1 - ε))) / (2ε)
    end
    @test g ≈ ref rtol = 1.0e-6
end

# ========================================
# Phase 1 — axis machinery + witness homogeneity
# ========================================

@testitem "Duck grid: Unitful axis machinery (internals)" begin
    using Unitful
    const FI = FastInterpolations

    su = 1.0u"s"
    Ts = typeof(su)

    @testset "Unitful range → generic _CachedRange (never index-space tags)" begin
        r = (0.0:0.5:4.0) .* u"s"
        cr = FI._to_float(r, Ts)
        @test cr isa FI._CachedRange{Ts}
        tag = typeof(cr).parameters[3]
        @test tag !== FI._UnitStep && tag !== FI._OneTo
        @test cr.h === 0.5u"s"
        @test cr.inv_h === inv(0.5u"s")     # Quantity⁻¹ — units preserved
        @test first(cr) === 0.0u"s" && last(cr) === 4.0u"s"
    end

    @testset "Unitful vector → _CachedVector (h/inv_h unit-typed)" begin
        xv = [0.0, 1.0, 2.5, 4.0] .* u"s"
        cv = FI._CachedVector(xv)
        @test eltype(cv.h) === Ts
        @test eltype(cv.inv_h) === typeof(inv(su))
        @test cv.h[2] === 1.5u"s"
    end

    @testset "demotion gate: Real index-space tags unchanged" begin
        @test typeof(FI._to_float(1:5, Float64)).parameters[3] === FI._UnitStep
        @test typeof(FI._to_float(Base.OneTo(5), Float64)).parameters[3] === FI._OneTo
        @test typeof(FI._to_float(1:5, Int)).parameters[3] === FI._UnitStep
    end
end

@testitem "Duck grid: witness homogeneity (Unitful promote_op)" begin
    using Unitful
    const FI = FastInterpolations

    Ts = typeof(1.0u"s")
    Tw = typeof(1.0u"W")

    # Dimensionally homogeneous witnesses must infer concrete unit-carrying
    # types (a non-homogeneous witness infers Union{} via DimensionError).
    @test Base.promote_op(FI._integrate_op, Ts, Tw, Ts) === typeof(1.0u"W*s")
    @test Base.promote_op(FI._coeff_op, Ts, Tw) === typeof(1.0u"W/s")     # order 1
    @test Base.promote_op(FI._coeff_op2, Ts, Tw) === typeof(1.0u"W/s^2") # order 2 (cubic z)
    @test Base.promote_op(FI._inv_op, Ts) === typeof(inv(1.0u"s"))
    @test Base.promote_op(FI._interp_op, Ts, Tw, Ts) === Tw              # eval: offset is dimensionless
end

# ========================================
# Phase 6 — error-quality pins (non-Real, non-supported grid types)
# ========================================

@testitem "Duck grid: error-quality pins (Date/String/wrong-unit)" begin
    using Dates
    using Unitful

    y = [0.0, 1.0, 0.5, 2.0, 1.0]

    @testset "Date grid: affine types are unsupported (loud, not silent)" begin
        # `Date - Date :: Day ≠ Date` breaks the closed-`-` duck contract
        # (cached spacing is stored at the coordinate type). Pin: loud error.
        xd = [Date(2020, 1, i) for i in 1:5]
        @test_throws Exception linear_interp(xd, y)
    end

    @testset "String grid: passes the isless guard, fails at arithmetic" begin
        xs = ["a", "b", "c", "d", "e"]
        # hasmethod(isless) is necessary-not-sufficient by design — the failure
        # is a loud arithmetic MethodError, never a silent wrong result.
        @test_throws Exception linear_interp(xs, y)
    end

    @testset "wrong-unit query: DimensionError surfaces" begin
        xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
        itp = linear_interp(xu, y .* u"W")
        @test_throws Unitful.DimensionError itp(1.5u"m")   # m vs s
    end
end

@testitem "Duck grid: Real type-identity insurance pins (review F10)" begin
    # The witness homogenization (value-space `_coeff_op`/`Tw` swaps) was
    # verified type-identical on Real inputs; these `===` pins freeze that so
    # a future witness edit that silently widens narrow/mixed Real combos
    # fails HERE, not in a downstream perf regression.
    x32 = Float32[0, 1, 2.5, 3, 4]
    y32 = Float32[1, 2, 4, 8, 5]
    y64 = Float64[1, 2, 4, 8, 5]

    @testset "narrow-type deriv1 stays narrow" begin
        @test typeof(linear_interp(x32, y32)(1.5f0; deriv = DerivOp(1))) === Float32
        @test typeof(cubic_interp(x32, y32)(1.5f0; deriv = DerivOp(1))) === Float32
    end

    @testset "local-slope families: cross-type widens to value precision" begin
        @test typeof(pchip_interp(x32, y64)(1.5f0)) === Float64
        @test typeof(pchip_interp(x32, y64)(1.5f0; deriv = DerivOp(1))) === Float64
        @test typeof(akima_interp(x32, y64)(1.5f0)) === Float64
        @test typeof(cardinal_interp(x32, y64)(1.5f0)) === Float64
    end

    @testset "ND 3D integrate out-fold types (`_integrate_nd_out_grids`)" begin
        g64 = (collect(0.0:1.0:2.0), collect(0.0:0.5:1.5), collect(0.0:0.25:0.75))
        g32 = map(g -> Float32.(g), g64)
        data = rand(3, 4, 4)
        @test typeof(integrate(interp(g64, data; method = LinearInterp()))) === Float64
        @test typeof(integrate(interp(g32, Float32.(data); method = LinearInterp()))) === Float32
        @test typeof(integrate(interp(g32, data; method = LinearInterp()))) === Float64
    end
end

@testitem "Number boundary: public coordinate/query params are `<:Number`" begin
    using Unitful

    # The public API relaxed `<:Real` to `<:Number` (numeric-coordinate admission
    # boundary), NOT to unbounded. A non-Number scalar query/grid must be rejected at
    # the boundary (no silent fall-through), while every Number coordinate — Real,
    # Unitful, Dual — is admitted. Guards this against re-loosening to `Any`, which
    # the `<:Real` lint cannot catch.
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 2.0, 4.0, 8.0]
    itp = linear_interp(x, y)

    @testset "non-Number query rejected (callable + one-shot)" begin
        @test_throws MethodError itp("nope")               # scalar callable is `xq::Number`
        @test_throws MethodError itp(["a", "b"])           # batch is `AbstractArray{<:Number}`
        @test_throws MethodError linear_interp(x, y, "nope")     # one-shot scalar query
        # `x0::Number` bounds skip the impl → the generic "not implemented" fallback.
        @test_throws Exception integrate(itp, "a", "b")         # integrate bounds are `::Number`
    end

    @testset "Number coordinates admitted (Real + Unitful)" begin
        @test itp(1.5) == 3.0
        @test integrate(itp, 0.5, 2.5) isa Number
        iu = linear_interp(x .* u"s", y .* u"W")
        @test iu(1.5u"s") == 3.0u"W"
        @test linear_interp(x .* u"s", y .* u"W", 1.5u"s") == 3.0u"W"
        @test integrate(iu, 0.5u"s", 2.5u"s") isa Unitful.Quantity
    end
end
