# Duck-typing tests for non-promotable Tv (e.g., SVector) crossed with non-
# promotable Tq (e.g., ForwardDiff.Dual).
#
# These cases hit `_output_eltype(Tv, Tg, Tq)`'s `Base.promote_op` fallback —
# `promote_type` can't model `SVector × Dual` and similar third-party chains.
# Tests pin the expected return types and container concreteness for both
# the persistent callable path and the one-shot path, plus type-stability
# (`@inferred`) and raw-eltype non-regression contracts.

@testitem "Persistent path — duck Tv × Tq widens via build + call" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "scalar Dual" begin
        @test (@inferred linear_interp(x, y_sv)(xq_d)) isa SVector{3, D}
        @test (@inferred cubic_interp(x, y_sv)(xq_d)) isa SVector{3, D}
        @test (@inferred quadratic_interp(x, y_sv)(xq_d)) isa SVector{3, D}
        @test (@inferred hermite_interp(x, y_sv, dy_sv)(xq_d)) isa SVector{3, D}
        @test (@inferred constant_interp(x, y_sv)(xq_d)) isa SVector{3, D}
    end

    @testset "batch Vector{Dual}" begin
        @test (@inferred linear_interp(x, y_sv)(xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred cubic_interp(x, y_sv)(xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred quadratic_interp(x, y_sv)(xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred hermite_interp(x, y_sv, dy_sv)(xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred constant_interp(x, y_sv)(xq_dv)) isa Vector{SVector{3, D}}
    end

    @testset "ND batch — SVector data × Tuple{Dual, Dual}" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
        q_dv = [
            (
                    ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0),
                    ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0),
                ) for i in 1:5
        ]
        @test (@inferred linear_interp((xg, yg), data_sv)(q_dv)) isa Vector{SVector{3, D}}
        @test (@inferred cubic_interp((xg, yg), data_sv)(q_dv)) isa Vector{SVector{3, D}}
        @test (@inferred constant_interp((xg, yg), data_sv)(q_dv)) isa Vector{SVector{3, D}}
    end

    @testset "ForwardDiff.derivative through indexed component (Issue #144 MWE)" begin
        for builder in (
                () -> linear_interp(x, y_sv),
                () -> cubic_interp(x, y_sv),
                () -> quadratic_interp(x, y_sv),
                () -> hermite_interp(x, y_sv, dy_sv),
                () -> constant_interp(x, y_sv),
            )
            itp = builder()
            d = ForwardDiff.derivative(t -> itp(t)[1], 2.5)
            @test d isa Float64
            @test isfinite(d)
        end
    end

    @testset "Plain Float64 y + Vector{Dual} batch (non-regression)" begin
        y = sin.(x)
        @test (@inferred linear_interp(x, y)(xq_dv)) isa Vector{D}
        @test (@inferred cubic_interp(x, y)(xq_dv)) isa Vector{D}
        @test (@inferred quadratic_interp(x, y)(xq_dv)) isa Vector{D}
        @test (@inferred hermite_interp(x, y, cos.(x))(xq_dv)) isa Vector{D}
        @test (@inferred constant_interp(x, y)(xq_dv)) isa Vector{D}
    end
end

@testitem "Oneshot path — duck Tv × Tq widens identically" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "1D scalar 3-arg" begin
        @test (@inferred linear_interp(x, y_sv, xq_d)) isa SVector{3, D}
        @test (@inferred cubic_interp(x, y_sv, xq_d)) isa SVector{3, D}
        @test (@inferred quadratic_interp(x, y_sv, xq_d)) isa SVector{3, D}
        @test (@inferred hermite_interp(x, y_sv, dy_sv, xq_d)) isa SVector{3, D}
        @test (@inferred constant_interp(x, y_sv, xq_d)) isa SVector{3, D}
    end

    @testset "1D batch 3-arg (the 4 cases broken before this fix)" begin
        @test (@inferred linear_interp(x, y_sv, xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred cubic_interp(x, y_sv, xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred quadratic_interp(x, y_sv, xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred hermite_interp(x, y_sv, dy_sv, xq_dv)) isa Vector{SVector{3, D}}
        @test (@inferred constant_interp(x, y_sv, xq_dv)) isa Vector{SVector{3, D}}
    end

    @testset "ND batch 3-arg" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
        q_dv = [
            (
                    ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0),
                    ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0),
                ) for i in 1:5
        ]
        @test (@inferred linear_interp((xg, yg), data_sv, q_dv)) isa Vector{SVector{3, D}}
        @test (@inferred cubic_interp((xg, yg), data_sv, q_dv)) isa Vector{SVector{3, D}}
        @test (@inferred quadratic_interp((xg, yg), data_sv, q_dv)) isa Vector{SVector{3, D}}
        # Constant ND oneshot batch — fixed in acfc95acf (was `::Vector{Tv}` typeassert).
        @test (@inferred constant_interp((xg, yg), data_sv, q_dv)) isa Vector{SVector{3, D}}
    end
end

@testitem "Series path — duck-Tv × duck-Tq carrier propagates" begin
    using StaticArrays, ForwardDiff
    using FastInterpolations: Series

    x = collect(1.0:10.0)
    y_sv1 = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    y_sv2 = [SA[-1.0 * i, 0.5i, 2.5i] for i in 1:10]
    s_sv = Series(y_sv1, y_sv2)
    xq_d = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "scalar Dual xq, SVector y per series" begin
        # Returns Vector{T_out} of length K = number of series.
        @test (@inferred linear_interp(x, s_sv, xq_d)) isa Vector{SVector{3, D}}
        @test (@inferred cubic_interp(x, s_sv, xq_d)) isa Vector{SVector{3, D}}
        @test (@inferred quadratic_interp(x, s_sv, xq_d)) isa Vector{SVector{3, D}}
        @test (@inferred constant_interp(x, s_sv, xq_d)) isa Vector{SVector{3, D}}
    end

    @testset "batch Vector{Dual} xq, SVector y per series" begin
        # Returns Vector{Vector{T_out}}, one inner vector per series.
        @test (@inferred linear_interp(x, s_sv, xq_dv)) isa Vector{Vector{SVector{3, D}}}
        @test (@inferred cubic_interp(x, s_sv, xq_dv)) isa Vector{Vector{SVector{3, D}}}
        @test (@inferred quadratic_interp(x, s_sv, xq_dv)) isa Vector{Vector{SVector{3, D}}}
        @test (@inferred constant_interp(x, s_sv, xq_dv)) isa Vector{Vector{SVector{3, D}}}
    end
end

@testitem "Adjoint path — duck-Tv y_bar widens correctly" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_for_slope = sin.(x)
    xq_for_adj = [2.5, 5.5, 8.5]
    y_bar_sv = [SA[0.1, 0.2, 0.3], SA[0.4, 0.5, 0.6], SA[0.7, 0.8, 0.9]]
    y_bar_dv = [ForwardDiff.Dual{Nothing}(0.1 + i, 1.0) for i in 1:3]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "1D — slope-free adjoints (Linear/Constant/Cubic/Quadratic)" begin
        for fn in (linear_adjoint, constant_adjoint, cubic_adjoint, quadratic_adjoint)
            @test (@inferred fn(x, xq_for_adj)(y_bar_sv)) isa Vector{SVector{3, Float64}}
            @test (@inferred fn(x, xq_for_adj)(y_bar_dv)) isa Vector{D}
        end
    end

    @testset "1D — Hermite-family adjoints" begin
        for fn in (hermite_adjoint, cardinal_adjoint)
            @test (@inferred fn(x, xq_for_adj)(y_bar_sv)) isa Vector{SVector{3, Float64}}
            @test (@inferred fn(x, xq_for_adj)(y_bar_dv)) isa Vector{D}
        end
        # Pchip/Akima carry y in the constructor (slope is data-dependent).
        for fn in (pchip_adjoint, akima_adjoint)
            @test (@inferred fn(x, y_for_slope, xq_for_adj)(y_bar_sv)) isa Vector{SVector{3, Float64}}
            @test (@inferred fn(x, y_for_slope, xq_for_adj)(y_bar_dv)) isa Vector{D}
        end
    end

    @testset "ND adjoints (Linear/Constant/Cubic/Quadratic)" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        xq_nd = [(2.5, 3.5), (3.0, 4.0), (4.0, 2.5)]
        for fn in (linear_adjoint, constant_adjoint, cubic_adjoint, quadratic_adjoint)
            @test (@inferred fn((xg, yg), xq_nd)(y_bar_sv)) isa Matrix{SVector{3, Float64}}
            @test (@inferred fn((xg, yg), xq_nd)(y_bar_dv)) isa Matrix{D}
        end
    end
end

@testitem "Type stability + raw-eltype contracts" begin
    using Test, StaticArrays, ForwardDiff
    using FastInterpolations: _output_eltype

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Constant scalar @inferred — no Union{Tv, Tq} return" begin
        # 1D: Int y, Float/Int xq — the kernel's `* one(dL)` and the right-edge
        # short-circuit must agree on the return type.
        x = 1:10
        y = (1:10) .^ 2
        itp = constant_interp(x, y)
        @test (@inferred itp(1.0)) === 1.0     # at first(x)
        @test (@inferred itp(10.0)) === 100.0  # at last(x): the short-circuit branch
        @test (@inferred itp(5.5)) === 25.0    # interior
        @test (@inferred itp(1)) === 1         # Int xq stays Int
        @test (@inferred itp(10)) === 100
        @test Base.return_types(itp, (Float64,)) == [Float64]
        @test Base.return_types(itp, (Int,)) == [Int]

        # ND: same contract through the anchor short-circuit path
        xg = 1:5; yg = 1:5
        data = [i + j for i in 1:5, j in 1:5]
        itp_nd = constant_interp((xg, yg), data)
        @test (@inferred itp_nd((1.0, 1.0))) isa Float64
        @test (@inferred itp_nd((1, 1))) isa Int
        @test Base.return_types(itp_nd, (Tuple{Float64, Float64},)) == [Float64]
        @test Base.return_types(itp_nd, (Tuple{Int, Int},)) == [Int]

        # Dual xq through Float y
        itp_f = constant_interp(1.0:10.0, collect(1.0:10.0))
        @test (@inferred itp_f(ForwardDiff.Dual{Nothing}(5.5, 1.0))) isa D
    end

    @testset "Constant Int → Int contract (scalar = batch consistency)" begin
        # Kernel `y * one(dL)` keeps Int×Int×Int chains in Int — trait now
        # routes through `_constant_kernel_shape`, so scalar/batch outputs
        # agree at the type level.
        x = collect(1:10)
        y = collect(10:10:100)
        @test constant_interp(x, y)(3) === 30
        @test (@inferred constant_interp(x, y)([2, 5, 8])) isa Vector{Int}
    end

    @testset "_output_eltype trait — type table" begin
        @test _output_eltype(Float64, Float64, Float64) === Float64
        @test _output_eltype(Int, Int, Int) === Float64                       # Int → Float upgrade
        @test _output_eltype(Int, Int, Float64) === Float64
        @test _output_eltype(Float32, Float64, Float64) === Float64
        @test _output_eltype(SVector{3, Float64}, Float64, Float64) === SVector{3, Float64}
        @test _output_eltype(SVector{3, Float64}, Float64, D) === SVector{3, D}         # Issue #144
        @test _output_eltype(SVector{3, Int}, Float64, Float64) === SVector{3, Float64} # silent Int-truncation fix
        @test _output_eltype(SVector{3, Int}, Float64, D) === SVector{3, D}
        @test _output_eltype(Complex{Int}, Float64, Float64) === ComplexF64
        @test _output_eltype(D, Float64, Float64) === D

        # Duck-type fallback: when `*` is undefined, return Tv unchanged.
        struct _DuckNoOp end
        @test _output_eltype(_DuckNoOp, Float64, Float64) === _DuckNoOp
    end
end

# Advanced AD shapes: Dual x grid, grid-offset sensitivity, nested Dual,
# ForwardDiff.gradient + Hessian. These pin cases beyond the basic
# {scalar, batch} × duck-Tv × duck-Tq matrix already covered above.
@testitem "Advanced AD — Dual grid, grid-offset, nested, gradient" begin
    using StaticArrays, ForwardDiff

    x_f = collect(1.0:10.0)
    y_f = sin.(x_f); dy_f = cos.(x_f)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    x_dg = [ForwardDiff.Dual{Nothing}(Float64(i), 1.0) for i in 1:10]
    xq_f = 5.5
    xq_d = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Dual x grid carrier flows through Tv × Tq" begin
        # Dual grid + Float query has latent Union{Float, Dual} inference for
        # some (fn, y) combos — runtime returns the right type, but `@inferred`
        # surfaces the Union. Plain `isa` until the inference is fixed.
        for fn in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            @test fn(x_dg, y_f)(xq_f) isa D
            @test fn(x_dg, y_f)(xq_d) isa D
            @test fn(x_dg, y_sv)(xq_f) isa SVector{3, D}
            @test fn(x_dg, y_sv)(xq_d) isa SVector{3, D}
            @test fn(x_dg, y_f, xq_d) isa D
            @test fn(x_dg, y_sv, xq_d) isa SVector{3, D}
        end
        @test hermite_interp(x_dg, y_f, dy_f)(xq_d) isa D
        @test hermite_interp(x_dg, y_sv, dy_sv)(xq_d) isa SVector{3, D}
    end

    @testset "ForwardDiff.derivative w.r.t. grid offset" begin
        # `∂/∂δ fn(x .+ δ, y, xq)` — grid sensitivity via AD. Persistent + one-shot.
        for fn in (linear_interp, cubic_interp, constant_interp)
            d_pers = ForwardDiff.derivative(δ -> fn(x_f .+ δ, y_f)(xq_f), 0.0)
            d_one = ForwardDiff.derivative(δ -> fn(x_f .+ δ, y_f, xq_f), 0.0)
            @test d_pers isa Float64 && isfinite(d_pers)
            @test d_one isa Float64 && isfinite(d_one)
            # SVector indexed component, one-shot path
            d_sv = ForwardDiff.derivative(δ -> fn(x_f .+ δ, y_sv, xq_f)[2], 0.0)
            @test d_sv isa Float64 && isfinite(d_sv)
        end
    end

    @testset "Nested Dual — 2nd-order AD (d²/dt²)" begin
        for fn in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            d2 = ForwardDiff.derivative(
                t1 -> ForwardDiff.derivative(t2 -> fn(x_f, y_f)(t2), t1),
                xq_f,
            )
            @test d2 isa Float64
        end
    end

    @testset "ND ForwardDiff.gradient + Hessian + grid-offset sensitivity" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data = [sin(i) * cos(j) for i in 1:5, j in 1:5]
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]

        # ForwardDiff.{gradient,hessian,derivative} have `Any` internal inference;
        # `@inferred` is over-strict for these AD wrappers.
        for fn in (linear_interp, cubic_interp)
            @test ForwardDiff.gradient(q -> fn((xg, yg), data)((q[1], q[2])), [2.5, 3.5]) isa Vector{Float64}
            @test ForwardDiff.hessian(q -> fn((xg, yg), data)((q[1], q[2])), [2.5, 3.5]) isa Matrix{Float64}
            @test ForwardDiff.gradient(q -> fn((xg, yg), data, (q[1], q[2])), [2.5, 3.5]) isa Vector{Float64}
            @test ForwardDiff.gradient(q -> fn((xg, yg), data_sv)((q[1], q[2]))[1], [2.5, 3.5]) isa Vector{Float64}
            @test ForwardDiff.gradient(q -> fn((xg, yg), data_sv, (q[1], q[2]))[1], [2.5, 3.5]) isa Vector{Float64}
            @test ForwardDiff.derivative(δ -> fn((xg .+ δ, yg), data, (2.5, 3.5)), 0.0) isa Float64
        end
    end

    @testset "Series + one-shot ForwardDiff.derivative on SVector component" begin
        using FastInterpolations: Series
        y_sv1 = [SA[Float64(i), 2.0i] for i in 1:10]
        y_sv2 = [SA[-1.0 * i, 0.5i] for i in 1:10]
        s_sv = Series(y_sv1, y_sv2)
        # `fn(x, series, t)` returns Vector{SVector} (one per series).
        # ∂/∂t of series[1][component[1]] exercises the full Series + duck-Tq chain.
        for fn in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            d = ForwardDiff.derivative(t -> fn(x_f, s_sv, t)[1][1], xq_f)
            @test d isa Float64
        end
    end
end

# `CubicSeriesInterpolant` build-then-call path — distinct from the 3-arg
# one-shot covered in the "Series path" testitem above.
@testitem "CubicSeriesInterpolant persistent path — SVector × Dual carrier" begin
    using StaticArrays, ForwardDiff
    using FastInterpolations: Series

    x = collect(1.0:10.0)
    y_sv1 = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    y_sv2 = [SA[-1.0 * i, 0.5i, 2.5i] for i in 1:10]
    s_sv = Series(y_sv1, y_sv2)
    xq_d = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    sitp = cubic_interp(x, s_sv)

    @testset "scalar Dual via persistent callable" begin
        @test (@inferred sitp(xq_d)) isa Vector{SVector{3, D}}
    end

    @testset "batch Vector{Dual} via persistent callable" begin
        @test (@inferred sitp(xq_dv)) isa Vector{Vector{SVector{3, D}}}
    end
end

# Mirror of the Cubic block above for the other three SeriesInterpolant
# variants. Each method's persistent callable allocates via its own kernel
# shape trait, so SVector × Dual must propagate without collapsing to
# Vector{Any} on any of them.
@testitem "Linear/Quadratic/Constant SeriesInterpolant persistent — SVector × Dual carrier" begin
    using StaticArrays, ForwardDiff
    using FastInterpolations: Series

    x = collect(1.0:10.0)
    y_sv1 = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    y_sv2 = [SA[-1.0 * i, 0.5i, 2.5i] for i in 1:10]
    s_sv = Series(y_sv1, y_sv2)
    xq_d = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Linear" begin
        sitp = linear_interp(x, s_sv)
        @test (@inferred sitp(xq_d)) isa Vector{SVector{3, D}}
        @test (@inferred sitp(xq_dv)) isa Vector{Vector{SVector{3, D}}}
    end

    @testset "Quadratic" begin
        sitp = quadratic_interp(x, s_sv)
        @test (@inferred sitp(xq_d)) isa Vector{SVector{3, D}}
        @test (@inferred sitp(xq_dv)) isa Vector{Vector{SVector{3, D}}}
    end

    @testset "Constant" begin
        sitp = constant_interp(x, s_sv)
        @test (@inferred sitp(xq_d)) isa Vector{SVector{3, D}}
        @test (@inferred sitp(xq_dv)) isa Vector{Vector{SVector{3, D}}}
    end
end

@testitem "Constant ND deriv-zero short-circuit propagates Tq carrier" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    xg = collect(1.0:5.0); yg = collect(1.0:5.0)
    data_int = [10i + j for i in 1:5, j in 1:5]
    qd = (ForwardDiff.Dual{Nothing}(2.5, 1.0), ForwardDiff.Dual{Nothing}(3.5, 0.0))

    @testset "Persistent ND scalar callable, deriv != EvalValue" begin
        itp = constant_interp((xg, yg), data_int)
        @test (@inferred itp(qd; deriv = (DerivOp(1), EvalValue()))) isa D
    end

    @testset "One-shot ND scalar callable, deriv != EvalValue" begin
        @test (@inferred constant_interp((xg, yg), data_int, qd; deriv = (DerivOp(1), EvalValue()))) isa D
    end
end

@testitem "Hermite one-shot preserves duck carrier in `dy`" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Float64 y + Vector{Dual} dy — batch one-shot" begin
        x = collect(1.0:10.0)
        y = sin.(x)
        dy = [ForwardDiff.Dual{Nothing}(cos(xi), 1.0) for xi in x]
        xq = collect(2.0:0.5:8.0)
        @test (@inferred hermite_interp(x, y, dy, xq)) isa Vector{D}
    end

    @testset "ForwardDiff.derivative on slope perturbation" begin
        x = collect(1.0:10.0)
        y = sin.(x)
        dy_base = cos.(x)
        xq = 5.5
        d = ForwardDiff.derivative(δ -> hermite_interp(x, y, dy_base .+ δ, [xq])[1], 0.0)
        @test d isa Float64
        @test isfinite(d)
    end

    @testset "Float32 y + Float64 dy precision pin" begin
        # Scalar/vector parity: wider `eltype(dy)` widens the result.
        x = Float32[0, 1, 2]
        y = Float32[0, 1, 4]
        dy = Float64[0, 2, 4]
        xq_v = Float32[0.5]
        @test eltype(hermite_interp(x, y, dy, xq_v)) === Float64
    end
end

# PCHIP/Cardinal/Akima oneshot allocators were switched to the kernel-shape
# trait via `_output_eltype(_arithmetic_kernel_shape, ...)`. Confirm the trait
# fires on the duck-Tq path (scalar + batch). PCHIP/Akima slope formulas use
# `sign`/`abs` on slopes so SVector y is not a supported carrier for those
# two; their `Tv` duck coverage is exercised via duck Tq on Float y.
# Cardinal's slopes are simple averages — it accepts SVector y, so we use the
# stronger SVector × Dual case there.
@testitem "Hermite-family one-shot — duck-Tq carrier through trait" begin
    using StaticArrays, ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    x = collect(1.0:10.0)
    y_f = sin.(x)
    xq_d = ForwardDiff.Dual{Nothing}(5.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.5i, 1.0) for i in 1:5]

    @testset "PCHIP — Float y + Dual xq" begin
        @test (@inferred pchip_interp(x, y_f, xq_d)) isa D
        @test (@inferred pchip_interp(x, y_f, xq_dv)) isa Vector{D}
    end

    @testset "Cardinal — SVector y + Dual xq" begin
        y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
        @test (@inferred cardinal_interp(x, y_sv, xq_d)) isa SVector{3, D}
        @test (@inferred cardinal_interp(x, y_sv, xq_dv)) isa Vector{SVector{3, D}}
    end

    @testset "Akima — Float y + Dual xq" begin
        @test (@inferred akima_interp(x, y_f, xq_d)) isa D
        @test (@inferred akima_interp(x, y_f, xq_dv)) isa Vector{D}
    end
end

# ND deriv-zero short-circuits multiply by `one(eltype(query[1]))` (axis-1).
# Heterogeneous-axis queries like `(Float, Dual)` lose the non-axis-1 carrier
# unless the short-circuit folds `one` over every axis (mirroring the forward
# kernel's per-axis `* one(dL_d)`).
@testitem "ND deriv-zero short-circuit per-axis carrier" begin
    using ForwardDiff

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    xg = collect(1.0:5.0); yg = collect(1.0:5.0)
    data_int = [10i + j for i in 1:5, j in 1:5]
    # Axis 1 Float, axis 2 Dual — Dual carrier lives on axis 2 only.
    q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0))
    q_het_batch = [q_het]

    @testset "Constant ND persistent scalar — (Float, Dual) × mixed deriv" begin
        itp = constant_interp((xg, yg), data_int)
        @test (@inferred itp(q_het; deriv = (EvalValue(), DerivOp(1)))) isa D
    end

    @testset "Constant ND one-shot scalar — (Float, Dual) × mixed deriv" begin
        @test (@inferred constant_interp((xg, yg), data_int, q_het; deriv = (EvalValue(), DerivOp(1)))) isa D
    end

    @testset "Constant ND one-shot batch — (Float, Dual) × mixed deriv" begin
        @test (@inferred constant_interp((xg, yg), data_int, q_het_batch; deriv = (EvalValue(), DerivOp(1)))) isa Vector{D}
    end

    @testset "Linear ND persistent scalar 2nd-deriv (zero-fill) — (Float, Dual)" begin
        # LinearInterpolantND _deriv_zero_fill triggers on 2nd-order deriv.
        data_f = [Float64(10i + j) for i in 1:5, j in 1:5]
        itp_l = linear_interp((xg, yg), data_f)
        @test (@inferred itp_l(q_het; deriv = (EvalValue(), DerivOp(2)))) isa D
    end
end

# Anchored-vector batch callable lets advanced users pre-build queries and
# reuse them across interpolants. Linear's path uses `_output_eltype(Tv, Tg, Tq)`
# for the output buffer; Constant + Quadratic only used `Vector{Tg}` so a
# duck-Tq anchor lost its carrier on `setindex!` convert.
@testitem "Anchored-vector callable propagates Tq carrier" begin
    using ForwardDiff
    using FastInterpolations: _ConstantAnchoredQuery, _QuadraticAnchoredQuery, _fill_anchors!

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    x = collect(0.0:0.1:1.0)
    y = sin.(x)
    xq_d = [ForwardDiff.Dual{Nothing}(0.15 + 0.1i, 1.0) for i in 0:4]

    @testset "Constant" begin
        itp = constant_interp(x, y)
        aq_vec = Vector{_ConstantAnchoredQuery{Float64, D}}(undef, length(xq_d))
        _fill_anchors!(aq_vec, x, xq_d, Val(:constant))
        @test (@inferred itp(aq_vec)) isa Vector{D}
    end

    @testset "Quadratic" begin
        itp = quadratic_interp(x, y)
        aq_vec = Vector{_QuadraticAnchoredQuery{Float64, D}}(undef, length(xq_d))
        _fill_anchors!(aq_vec, x, xq_d, Val(:quadratic))
        @test (@inferred itp(aq_vec)) isa Vector{D}
    end

    # Int-chain preservation: ConstantInterpolant's kernel is `yv * one(q)`,
    # so a fully-Int chain (`Tv = Int, Tq = Int`) must stay Int through the
    # anchored callable. The legacy 2-arg `_output_eltype` Float-upgraded any
    # `Tr <: _PromotableValue && !<:AbstractFloat`, producing Vector{Float64}
    # for the same input — the scalar/oneshot paths already preserved Int.
    @testset "Constant Int-chain preservation" begin
        x_int = collect(0:5)
        y_int = x_int .* x_int
        itp = constant_interp(x_int, y_int)
        xq_int = [1, 2, 3]
        aq_vec = Vector{_ConstantAnchoredQuery{Int, Int}}(undef, length(xq_int))
        _fill_anchors!(aq_vec, x_int, xq_int, Val(:constant))
        out = itp(aq_vec)
        @test out isa Vector{Int}
        @test out == [itp(xq) for xq in xq_int]
    end
end

# Inference-quality contract: every duck path (SVector × Dual, Float y + Dual
# dy, heterogeneous axis carrier, anchored-vector callable, …) must be
# type-stable. `@inferred` errors if the inferred return type doesn't match
# the actual result type — catches silent Union returns and `Any` fallbacks
# that `isa` checks would miss.
@testitem "Type stability — duck-Tv × duck-Tq inference" begin
    using StaticArrays, ForwardDiff
    using FastInterpolations: Series

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    SVD = SVector{3, D}

    @testset "1D persistent — scalar Dual" begin
        let l = linear_interp(x, y_sv), c = cubic_interp(x, y_sv),
                q = quadratic_interp(x, y_sv), k = constant_interp(x, y_sv),
                h = hermite_interp(x, y_sv, dy_sv)
            @test (@inferred l(xq_d)) isa SVD
            @test (@inferred c(xq_d)) isa SVD
            @test (@inferred q(xq_d)) isa SVD
            @test (@inferred k(xq_d)) isa SVD
            @test (@inferred h(xq_d)) isa SVD
        end
    end

    @testset "1D persistent — batch Vector{Dual}" begin
        let l = linear_interp(x, y_sv), c = cubic_interp(x, y_sv),
                q = quadratic_interp(x, y_sv), k = constant_interp(x, y_sv),
                h = hermite_interp(x, y_sv, dy_sv)
            @test (@inferred l(xq_dv)) isa Vector{SVD}
            @test (@inferred c(xq_dv)) isa Vector{SVD}
            @test (@inferred q(xq_dv)) isa Vector{SVD}
            @test (@inferred k(xq_dv)) isa Vector{SVD}
            @test (@inferred h(xq_dv)) isa Vector{SVD}
        end
    end

    @testset "1D oneshot 3-arg — scalar Dual" begin
        @test (@inferred linear_interp(x, y_sv, xq_d)) isa SVD
        @test (@inferred cubic_interp(x, y_sv, xq_d)) isa SVD
        @test (@inferred quadratic_interp(x, y_sv, xq_d)) isa SVD
        @test (@inferred constant_interp(x, y_sv, xq_d)) isa SVD
        @test (@inferred hermite_interp(x, y_sv, dy_sv, xq_d)) isa SVD
    end

    @testset "1D oneshot 3-arg — batch Vector{Dual}" begin
        @test (@inferred linear_interp(x, y_sv, xq_dv)) isa Vector{SVD}
        @test (@inferred cubic_interp(x, y_sv, xq_dv)) isa Vector{SVD}
        @test (@inferred quadratic_interp(x, y_sv, xq_dv)) isa Vector{SVD}
        @test (@inferred constant_interp(x, y_sv, xq_dv)) isa Vector{SVD}
        @test (@inferred hermite_interp(x, y_sv, dy_sv, xq_dv)) isa Vector{SVD}
    end

    @testset "ND batch — SVector data × (Dual, Dual)" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
        q_dv = [
            (
                    ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0),
                    ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0),
                ) for i in 1:5
        ]
        let l = linear_interp((xg, yg), data_sv), c = cubic_interp((xg, yg), data_sv),
                k = constant_interp((xg, yg), data_sv)
            @test (@inferred l(q_dv)) isa Vector{SVD}
            @test (@inferred c(q_dv)) isa Vector{SVD}
            @test (@inferred k(q_dv)) isa Vector{SVD}
        end
        @test (@inferred linear_interp((xg, yg), data_sv, q_dv)) isa Vector{SVD}
        @test (@inferred cubic_interp((xg, yg), data_sv, q_dv)) isa Vector{SVD}
        @test (@inferred constant_interp((xg, yg), data_sv, q_dv)) isa Vector{SVD}
    end

    @testset "Constant Int chain — scalar/batch type-stable Int" begin
        # Kernel-shape trait: `_constant_kernel_shape(xL, yv, xq) = yv * one(xq - xL)`
        # → Int×Int×Int inferred as Int (no spurious Float upgrade).
        xi = collect(1:10); yi = collect(10:10:100)
        let k = constant_interp(xi, yi)
            @test (@inferred k(3)) === 30
            @test (@inferred k([2, 5, 8])) isa Vector{Int}
        end
        @test (@inferred constant_interp(xi, yi, [2, 5, 8])) isa Vector{Int}
    end

    @testset "ND deriv-zero short-circuit — heterogeneous (Float, Dual)" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_int = [10i + j for i in 1:5, j in 1:5]
        q_het = (2.5, ForwardDiff.Dual{Nothing}(3.5, 1.0))
        let k = constant_interp((xg, yg), data_int)
            @test (@inferred k(q_het; deriv = (EvalValue(), DerivOp(1)))) isa D
        end
    end

    @testset "Hermite Float y + Dual dy precision carrier" begin
        xs = collect(1.0:10.0)
        ys = sin.(xs)
        dys = [ForwardDiff.Dual{Nothing}(cos(xi), 1.0) for xi in xs]
        xqs = collect(2.0:0.5:8.0)
        @test (@inferred hermite_interp(xs, ys, dys, xqs)) isa Vector{D}
    end

    @testset "CubicSeriesInterpolant persistent — SVector × Dual" begin
        y_sv1 = [SA[Float64(i), 2.0i] for i in 1:10]
        y_sv2 = [SA[-1.0 * i, 0.5i] for i in 1:10]
        s_sv = Series(y_sv1, y_sv2)
        sitp = cubic_interp(x, s_sv)
        @test (@inferred sitp(xq_d)) isa Vector{SVector{2, D}}
        @test (@inferred sitp(xq_dv)) isa Vector{Vector{SVector{2, D}}}
    end

    @testset "Anchored-vector callable — Constant + Quadratic × Dual" begin
        using FastInterpolations: _ConstantAnchoredQuery, _QuadraticAnchoredQuery, _fill_anchors!
        xs = collect(0.0:0.1:1.0)
        ys = sin.(xs)
        xq_dual = [ForwardDiff.Dual{Nothing}(0.15 + 0.1i, 1.0) for i in 0:4]
        let itp_c = constant_interp(xs, ys), itp_q = quadratic_interp(xs, ys)
            aq_c = Vector{_ConstantAnchoredQuery{Float64, D}}(undef, length(xq_dual))
            _fill_anchors!(aq_c, xs, xq_dual, Val(:constant))
            @test (@inferred itp_c(aq_c)) isa Vector{D}
            aq_q = Vector{_QuadraticAnchoredQuery{Float64, D}}(undef, length(xq_dual))
            _fill_anchors!(aq_q, xs, xq_dual, Val(:quadratic))
            @test (@inferred itp_q(aq_q)) isa Vector{D}
        end
    end
end

# Rational chains: `Rational / Rational = Rational` in Julia (no Float upgrade),
# so the arithmetic kernel `y0 + (y1-y0) * (dL/h)` actually preserves Rational.
# The kernel-shape trait predicts this exactly via `Base.promote_op`. The old
# `_PromotableValue` enumeration over-predicted Float (because `Rational <:
# _PromotableValue` triggered `float(...)`).
@testitem "Kernel-shape trait pins Rational chain (no spurious Float upgrade)" begin
    using FastInterpolations: _arithmetic_kernel_shape, _constant_kernel_shape, _output_eltype

    @testset "Trait-level — `@inferred` type table" begin
        # Arithmetic shape: `Rational + Rational * (Rational/Rational) = Rational`.
        @test (@inferred _output_eltype(_arithmetic_kernel_shape, Rational{Int}, Rational{Int}, Rational{Int})) === Rational{Int}
        # Sanity: Int/Float baselines match the previous trait.
        @test (@inferred _output_eltype(_arithmetic_kernel_shape, Int, Int, Int)) === Float64
        @test (@inferred _output_eltype(_arithmetic_kernel_shape, Float64, Float64, Float64)) === Float64
        @test (@inferred _output_eltype(_arithmetic_kernel_shape, Int, Complex{Int}, Int)) === ComplexF64
        # Selection shape: `Rational * one(Rational - Rational) = Rational`.
        @test (@inferred _output_eltype(_constant_kernel_shape, Rational{Int}, Rational{Int}, Rational{Int})) === Rational{Int}
        @test (@inferred _output_eltype(_constant_kernel_shape, Int, Int, Int)) === Int
    end

    @testset "Constant — end-to-end Rational preserved (storage stays raw)" begin
        # Constant 1D + ND store `{Tg, Tv}` directly (no `_promote_grid_float`),
        # so the kernel-shape trait's Rational prediction flows through to the
        # user-visible Vector eltype.
        x_r = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1]
        y_r = Rational{Int}[1 // 2, 3 // 2, 5 // 2, 7 // 2]
        xq_r = 3 // 2
        xq_r_vec = Rational{Int}[1 // 2, 3 // 2, 5 // 2]
        let itp = constant_interp(x_r, y_r)
            @test (@inferred itp(xq_r)) isa Rational{Int}
            @test (@inferred itp(xq_r_vec)) isa Vector{Rational{Int}}
        end
        @test (@inferred constant_interp(x_r, y_r, xq_r_vec)) isa Vector{Rational{Int}}
    end

    @testset "Linear — Rational user-visible blocked by `_promote_grid_float` (BROKEN)" begin
        # The arithmetic-kernel-shape trait correctly predicts Rational for a
        # fully-Rational chain, but `LinearInterpolant`'s constructor calls
        # `_promote_grid_float` which lifts the grid/values to Float64 at build
        # time — the trait's Rational prediction is shadowed by storage Float64.
        # Pinned as `@test_broken` so a future relaxation of that lift will
        # auto-surface as a `now passing` alert.
        x_r = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1]
        y_r = Rational{Int}[1 // 2, 3 // 2, 5 // 2, 7 // 2]
        xq_r = 3 // 2
        xq_r_vec = Rational{Int}[1 // 2, 3 // 2, 5 // 2]
        @test_broken linear_interp(x_r, y_r)(xq_r) isa Rational{Int}
        @test_broken linear_interp(x_r, y_r)(xq_r_vec) isa Vector{Rational{Int}}
        @test_broken linear_interp(x_r, y_r, xq_r_vec) isa Vector{Rational{Int}}
    end

    # Cubic/Quadratic and the Hermite family (PCHIP/Cardinal/Akima) hit the
    # SAME barrier as Linear plus a second one: their internal coefficient
    # eltype (`Tz` for cubic z, `Tc`/`Tcoeff` for quadratic a/d, slope
    # coefficients for Hermite-family) is sized via the legacy 2-arg
    # `_output_eltype(Tv, Tg)` which also Float-forces. Both barriers must
    # relax before Rational reaches the user; until then these pin BROKEN.
    # Use enough points to satisfy each method's minimum (Akima needs ≥5).
    @testset "Cubic — Rational user-visible blocked (BROKEN)" begin
        # Persistent path hits both barriers (storage + coefficient lift) → BROKEN.
        #
        # The 3-arg oneshot's `isa Vector{Rational{Int}}` check would PASS but
        # for the wrong reason: the kernel-shape trait correctly sizes the
        # output buffer as `Vector{Rational{Int}}`, but the in-place kernel
        # still computes in Float64 internally (storage lift) and writes
        # results via bit-exact `convert(Rational{Int}, ::Float64)`. The user
        # gets Float64-bits-as-Rational, not true Rational arithmetic.
        #
        # We pin the SEMANTIC contract instead: a cubic spline of `y = x`
        # should reproduce `y = x` exactly. Under pure Rational arithmetic
        # (post-follow-up), `cubic_interp(x, x, [3//7])` returns `[3//7]`.
        # Under current Float64-internal computation it returns the bit-exact
        # Rational form of `Float64(3/7)` (denominator ~2^52). The semantic
        # equality fails today and auto-surfaces when the storage-lift fix lands.
        x_r = Rational{Int}[i // 1 for i in 0:9]
        y_r = Rational{Int}[(i * i) // 2 for i in 0:9]
        xq_r = 9 // 4
        xq_r_vec = Rational{Int}[(i + 1) // 2 for i in 0:5]
        @test_broken cubic_interp(x_r, y_r)(xq_r) isa Rational{Int}
        @test_broken cubic_interp(x_r, y_r)(xq_r_vec) isa Vector{Rational{Int}}
        # Semantic pin: y=x cubic spline must reproduce y=x exactly.
        y_linear = copy(x_r)
        xq_nondyadic = Rational{Int}[1 // 7, 2 // 7, 3 // 7]
        @test_broken cubic_interp(x_r, y_linear, xq_nondyadic) == xq_nondyadic
    end

    @testset "Quadratic — Rational user-visible blocked (BROKEN)" begin
        x_r = Rational{Int}[i // 1 for i in 0:9]
        y_r = Rational{Int}[(i * i) // 2 for i in 0:9]
        xq_r = 9 // 4
        xq_r_vec = Rational{Int}[(i + 1) // 2 for i in 0:5]
        @test_broken quadratic_interp(x_r, y_r)(xq_r) isa Rational{Int}
        @test_broken quadratic_interp(x_r, y_r)(xq_r_vec) isa Vector{Rational{Int}}
        @test_broken quadratic_interp(x_r, y_r, xq_r_vec) isa Vector{Rational{Int}}
    end

    @testset "Hermite-family — Rational user-visible blocked (BROKEN)" begin
        x_r = Rational{Int}[i // 1 for i in 0:9]
        y_r = Rational{Int}[(i * i) // 2 for i in 0:9]
        xq_r = 9 // 4
        xq_r_vec = Rational{Int}[(i + 1) // 2 for i in 0:5]
        @test_broken pchip_interp(x_r, y_r, xq_r_vec) isa Vector{Rational{Int}}
        @test_broken cardinal_interp(x_r, y_r, xq_r_vec) isa Vector{Rational{Int}}
        @test_broken akima_interp(x_r, y_r, xq_r_vec) isa Vector{Rational{Int}}
    end
end

# PCHIP/Akima impose NO `<: AbstractFloat` on `Tv` — only `Tq <: Real`.
# Their slope formulas demand an interface (`sign`, `abs`, `zero`, `+`, `-`,
# `*`, `/`, ordering) but accept any duck type that implements it. The
# SVector failure isn't a method restriction — it's that StaticArrays
# doesn't define `sign`/`abs` on `SVector`. A minimal scalar-wrapper duck
# that implements the interface flows through cleanly and the trait
# predicts the wrapper type as the kernel return.
@testitem "PCHIP/Akima — custom scalar duck-Tv interface" begin
    # Wrapper around Float64 implementing the slope-formula interface.
    struct DuckFloat
        v::Float64
    end
    Base.sign(d::DuckFloat) = DuckFloat(sign(d.v))
    Base.abs(d::DuckFloat) = DuckFloat(abs(d.v))
    Base.zero(::Type{DuckFloat}) = DuckFloat(0.0)
    Base.zero(::DuckFloat) = DuckFloat(0.0)
    Base.:-(a::DuckFloat) = DuckFloat(-a.v)
    Base.:+(a::DuckFloat, b::DuckFloat) = DuckFloat(a.v + b.v)
    Base.:-(a::DuckFloat, b::DuckFloat) = DuckFloat(a.v - b.v)
    Base.:*(a::DuckFloat, b::DuckFloat) = DuckFloat(a.v * b.v)
    Base.:/(a::DuckFloat, b::DuckFloat) = DuckFloat(a.v / b.v)
    Base.:*(a::DuckFloat, b::Real) = DuckFloat(a.v * b)
    Base.:*(b::Real, a::DuckFloat) = DuckFloat(b * a.v)
    Base.:/(a::DuckFloat, b::Real) = DuckFloat(a.v / b)
    Base.:/(a::Real, b::DuckFloat) = DuckFloat(a / b.v)
    Base.:+(a::Real, b::DuckFloat) = DuckFloat(a + b.v)
    Base.:+(a::DuckFloat, b::Real) = DuckFloat(a.v + b)
    Base.:-(a::Real, b::DuckFloat) = DuckFloat(a - b.v)
    Base.:-(a::DuckFloat, b::Real) = DuckFloat(a.v - b)
    Base.isless(a::DuckFloat, b::DuckFloat) = a.v < b.v
    Base.:(==)(a::DuckFloat, b::DuckFloat) = a.v == b.v

    x = collect(1.0:10.0)
    y_d = [DuckFloat(sin(xi)) for xi in x]
    xq_scalar = 5.5
    xq_batch = [2.5, 4.5, 6.5, 8.5]

    @testset "PCHIP" begin
        @test (@inferred pchip_interp(x, y_d, xq_scalar)) isa DuckFloat
        @test (@inferred pchip_interp(x, y_d, xq_batch)) isa Vector{DuckFloat}
    end

    @testset "Akima" begin
        @test (@inferred akima_interp(x, y_d, xq_scalar)) isa DuckFloat
        @test (@inferred akima_interp(x, y_d, xq_batch)) isa Vector{DuckFloat}
    end
end

# ============================================================================
# OOB carrier + empty-path eltype contracts (PR #146 Copilot/Codex reviews)
# ============================================================================
# Existing duck-Tv tests above only exercise in-domain queries, so the OOB
# branches (`_eval_extrapolation` → `_promote_extrap_val`, `_try_fill_oob` →
# `_fill_extrap_result`) never run with non-Number Tv. The fallback
# `_promote_extrap_val(val, xq) = val` drops the Tq carrier for `SVector` /
# non-Number Tv, so scalar OOB returns raw `SVector{Float64}` while batch
# OOB returns `SVector{Dual}` via the trait-sized buffer — scalar/batch
# disagree. Pinned as `@test_broken`; a follow-up fix to
# `_promote_extrap_val`/`_promote_extrap_zero` will flip these to `@test`.
@testitem "OOB carrier — scalar/batch consistency for non-Number Tv" begin
    using StaticArrays, ForwardDiff
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_sv = [SA[Float64(i), 2.0i] for i in 1:5]
    xq_in = ForwardDiff.Dual{Nothing}(1.5, 1.0)
    xq_oob_lo = ForwardDiff.Dual{Nothing}(-1.0, 1.0)
    xq_oob_hi = ForwardDiff.Dual{Nothing}(10.0, 1.0)

    @testset "1D scalar OOB ClampExtrap — SVector y + Dual xq" begin
        for method in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            itp = method(x, y_sv; extrap = ClampExtrap())
            T_in = typeof(itp(xq_in))
            @test typeof(itp(xq_oob_lo)) === T_in
            @test typeof(itp(xq_oob_hi)) === T_in
            # scalar/batch agreement at OOB
            @test typeof(itp([xq_oob_lo])[1]) === typeof(itp(xq_oob_lo))
            @test typeof(itp([xq_oob_hi])[1]) === typeof(itp(xq_oob_hi))
        end
    end

    @testset "1D scalar OOB FillExtrap — SVector fill + Dual xq" begin
        fill_v = SA[99.0, 99.0]
        for method in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            itp = method(x, y_sv; extrap = FillExtrap(fill_v))
            T_in = typeof(itp(xq_in))
            @test typeof(itp(xq_oob_lo)) === T_in
            @test typeof(itp([xq_oob_lo])[1]) === typeof(itp(xq_oob_lo))
        end
    end

    @testset "ND scalar OOB FillExtrap — SVector data/fill + Dual queries" begin
        # 5-point grids to satisfy cubic CubicFit BC defaults.
        xg = collect(0.0:1.0:4.0); yg = collect(0.0:1.0:4.0)
        data_sv = [SA[Float64(i), Float64(j)] for i in 1:5, j in 1:5]
        fill_v = SA[99.0, 99.0]
        q_in_t = (
            ForwardDiff.Dual{Nothing}(1.5, 1.0),
            ForwardDiff.Dual{Nothing}(1.5, 1.0),
        )
        q_oob_t = (
            ForwardDiff.Dual{Nothing}(-1.0, 1.0),
            ForwardDiff.Dual{Nothing}(-1.0, 1.0),
        )
        for method in (linear_interp, cubic_interp, constant_interp)
            itp = method((xg, yg), data_sv; extrap = FillExtrap(fill_v))
            T_in = typeof(itp(q_in_t))
            @test typeof(itp(q_oob_t)) === T_in
            @test typeof(itp([q_oob_t])[1]) === typeof(itp(q_oob_t))
        end
    end

    @testset "ND scalar OOB ClampExtrap — SVector data + Dual queries" begin
        xg = collect(0.0:1.0:4.0); yg = collect(0.0:1.0:4.0)
        data_sv = [SA[Float64(i), Float64(j)] for i in 1:5, j in 1:5]
        q_in_t = (
            ForwardDiff.Dual{Nothing}(1.5, 1.0),
            ForwardDiff.Dual{Nothing}(1.5, 1.0),
        )
        q_oob_t = (
            ForwardDiff.Dual{Nothing}(-1.0, 1.0),
            ForwardDiff.Dual{Nothing}(-1.0, 1.0),
        )
        for method in (linear_interp, cubic_interp, constant_interp)
            itp = method((xg, yg), data_sv; extrap = ClampExtrap())
            T_in = typeof(itp(q_in_t))
            @test typeof(itp(q_oob_t)) === T_in
            @test typeof(itp([q_oob_t])[1]) === typeof(itp(q_oob_t))
        end
    end
end

# Constant ND oneshot derivative fast path: `nq == 0 && return Vector{Tv}(undef, 0)`
# ignores the query carrier, while the `nq > 0` branch synthesises
# `0 * first(data) * prod(one, first_q)` (carrier-aware). Empty and non-empty
# results disagree on eltype — pinned `@test_broken` until the empty branch
# allocates with the same trait-derived eltype as the non-empty branch.
@testitem "ND oneshot derivative — empty vs non-empty query eltype" begin
    using ForwardDiff
    xg = [0.0, 1.0, 2.0]; yg = [0.0, 1.0, 2.0]
    data = Float64[10i + j for i in 1:3, j in 1:3]
    q_one_d = [
        (
            ForwardDiff.Dual{Nothing}(0.5, 1.0),
            ForwardDiff.Dual{Nothing}(0.5, 1.0),
        ),
    ]
    q_empty_d = typeof(q_one_d)()
    derivs = (EvalValue(), DerivOp(1))
    r_one = constant_interp((xg, yg), data, q_one_d; deriv = derivs)
    r_empty = constant_interp((xg, yg), data, q_empty_d; deriv = derivs)
    @test eltype(r_one) === eltype(r_empty)
    @test length(r_empty) == 0
end
