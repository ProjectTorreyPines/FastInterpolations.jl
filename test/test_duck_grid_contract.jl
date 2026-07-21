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
        @test Base.promote_op(FI._integrate_op, T, T, T) === R
        @test Base.promote_op(FI._inv_op, T) === R
    end

    @testset "mixed combos" begin
        @test Base.promote_op(FI._integrate_op, Int, Float64, Int) === Float64
        @test Base.promote_op(FI._interp_op, Float32, Float64, Float32) === Float64
        # inv(Int)::Float64 dominates Float32 values — current (pinned) behavior.
        @test Base.promote_op(FI._coeff_op, Int, Float32) === Float64
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
