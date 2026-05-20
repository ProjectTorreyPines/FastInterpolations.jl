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
        @test linear_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test cubic_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test quadratic_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test hermite_interp(x, y_sv, dy_sv)(xq_d) isa SVector{3, D}
        @test constant_interp(x, y_sv)(xq_d) isa SVector{3, D}
    end

    @testset "batch Vector{Dual}" begin
        @test linear_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test cubic_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test quadratic_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test hermite_interp(x, y_sv, dy_sv)(xq_dv) isa Vector{SVector{3, D}}
        @test constant_interp(x, y_sv)(xq_dv) isa Vector{SVector{3, D}}
    end

    @testset "ND batch — SVector data × Tuple{Dual, Dual}" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
        q_dv = [(ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0),
                 ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0)) for i in 1:5]
        @test linear_interp((xg, yg), data_sv)(q_dv) isa Vector{SVector{3, D}}
        @test cubic_interp((xg, yg), data_sv)(q_dv) isa Vector{SVector{3, D}}
        @test constant_interp((xg, yg), data_sv)(q_dv) isa Vector{SVector{3, D}}
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
        @test linear_interp(x, y)(xq_dv) isa Vector{D}
        @test cubic_interp(x, y)(xq_dv) isa Vector{D}
        @test quadratic_interp(x, y)(xq_dv) isa Vector{D}
        @test hermite_interp(x, y, cos.(x))(xq_dv) isa Vector{D}
        @test constant_interp(x, y)(xq_dv) isa Vector{D}
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
        @test linear_interp(x, y_sv, xq_d) isa SVector{3, D}
        @test cubic_interp(x, y_sv, xq_d) isa SVector{3, D}
        @test quadratic_interp(x, y_sv, xq_d) isa SVector{3, D}
        @test hermite_interp(x, y_sv, dy_sv, xq_d) isa SVector{3, D}
        @test constant_interp(x, y_sv, xq_d) isa SVector{3, D}
    end

    @testset "1D batch 3-arg (the 4 cases broken before this fix)" begin
        @test linear_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
        @test cubic_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
        @test quadratic_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
        @test hermite_interp(x, y_sv, dy_sv, xq_dv) isa Vector{SVector{3, D}}
        @test constant_interp(x, y_sv, xq_dv) isa Vector{SVector{3, D}}
    end

    @testset "ND batch 3-arg (latent — same root cause)" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
        q_dv = [(ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0),
                 ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0)) for i in 1:5]
        @test linear_interp((xg, yg), data_sv, q_dv) isa Vector{SVector{3, D}}
        @test cubic_interp((xg, yg), data_sv, q_dv) isa Vector{SVector{3, D}}
        @test quadratic_interp((xg, yg), data_sv, q_dv) isa Vector{SVector{3, D}}
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
            @test fn(x, xq_for_adj)(y_bar_sv) isa Vector{SVector{3, Float64}}
            @test fn(x, xq_for_adj)(y_bar_dv) isa Vector{D}
        end
    end

    @testset "1D — Hermite-family adjoints" begin
        for fn in (hermite_adjoint, cardinal_adjoint)
            @test fn(x, xq_for_adj)(y_bar_sv) isa Vector{SVector{3, Float64}}
            @test fn(x, xq_for_adj)(y_bar_dv) isa Vector{D}
        end
        # Pchip/Akima carry y in the constructor (slope is data-dependent).
        for fn in (pchip_adjoint, akima_adjoint)
            @test fn(x, y_for_slope, xq_for_adj)(y_bar_sv) isa Vector{SVector{3, Float64}}
            @test fn(x, y_for_slope, xq_for_adj)(y_bar_dv) isa Vector{D}
        end
    end

    @testset "ND adjoints (Linear/Constant/Cubic/Quadratic)" begin
        xg = collect(1.0:5.0); yg = collect(1.0:5.0)
        xq_nd = [(2.5, 3.5), (3.0, 4.0), (4.0, 2.5)]
        for fn in (linear_adjoint, constant_adjoint, cubic_adjoint, quadratic_adjoint)
            @test fn((xg, yg), xq_nd)(y_bar_sv) isa Matrix{SVector{3, Float64}}
            @test fn((xg, yg), xq_nd)(y_bar_dv) isa Matrix{D}
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
        y = (1:10).^2
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

    @testset "Constant Int → Int contract (non-regression)" begin
        x = collect(1:10)
        y = collect(10:10:100)
        @test constant_interp(x, y)(3) isa Int
        @test constant_interp(x, y)([2, 5, 8]) isa Vector{Int}
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
