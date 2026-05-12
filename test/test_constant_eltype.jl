# Tests for constant_interp eltype duck-type policy (1D).
#
# Constant is a pure selection kernel (`y_left` for LeftSide, `dL <= h/2 ? y_left
# : y_right` for NearestSide) — no x·y arithmetic. The output contract therefore
# follows `eltype(y)` rather than the eager Float widening used by arithmetic
# methods (Linear/Cubic/…). These tests pin that contract end-to-end across
# persistent, oneshot, series, periodic, adjoint, and show paths.
#
# Range Int grids land on `_CachedRange{Int, Float64}` via the `Tinv`-aware
# `_to_float` (mirroring `_CachedVector{T, Tinv}`).

# ============================================================================
# Group 1: Forward paths (persistent / oneshot / series / periodic / show)
# ============================================================================
@testitem "Constant eltype duck-type — forward" begin
    import FastInterpolations: ConstantInterpolant, ConstantSeriesInterpolant,
        _CachedRange, _CachedVector, _ExclusivePeriodicAxis

    @testset "Persistent constructor (1D)" begin
        @testset "Int Vector x, Int y → output Int" begin
            x = [0, 1, 2, 3, 4]
            y = [10, 20, 30, 40, 50]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Int, Int}
            @test eltype(itp.y) === Int
            @test itp.x isa _CachedVector{Int, Float64}
            @test itp(1.5) isa Int
            @test itp(1.5) == 20
        end

        @testset "Int Range x, Int y → itp.x::_CachedRange{Int, Float64}" begin
            x = 0:1:4
            y = [10, 20, 30, 40, 50]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Int, Int}
            @test itp.x isa _CachedRange{Int, Float64}
            @test itp.x.h === 1
            @test itp.x.inv_h === 1.0
            @test itp(2.5) isa Int
            @test itp(2.5) == 30
        end

        @testset "Rational x, Rational y → output Rational{Int}" begin
            x = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1, 4 // 1]
            y = Rational{Int}[1 // 2, 3 // 2, 5 // 2, 7 // 2, 9 // 2]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Rational{Int}, Rational{Int}}
            @test itp(3 // 2) isa Rational{Int}
            @test itp(3 // 2) === 3 // 2
        end

        @testset "Float64 x, Int y → output Int (NEW: no Float widening)" begin
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = [10, 20, 30, 40, 50]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Float64, Int}
            @test itp(1.5) isa Int
            @test itp(1.5) == 20
        end

        @testset "Float32 x, Float32 y → output Float32 (regression guard)" begin
            x = Float32[0.0, 1.0, 2.0, 3.0, 4.0]
            y = Float32[10.0, 20.0, 30.0, 40.0, 50.0]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Float32, Float32}
            @test itp(1.5f0) isa Float32
        end

        @testset "Complex y preserved" begin
            x = [0.0, 1.0, 2.0]
            y = ComplexF64[1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Float64, ComplexF64}
            @test itp(0.5) isa ComplexF64
            @test itp(0.5) === 1.0 + 2.0im
        end
    end

    @testset "Oneshot (scalar / vector-alloc / in-place)" begin
        x_int = [0, 1, 2, 3, 4]
        y_int = [10, 20, 30, 40, 50]

        @testset "scalar oneshot: Int input → Int output" begin
            r = constant_interp(x_int, y_int, 2)
            @test r isa Int
            @test r == 30
        end

        @testset "vector alloc oneshot: Vector{Int} 결과" begin
            v = constant_interp(x_int, y_int, [0, 1, 2, 3])
            @test v isa Vector{Int}
            @test v == [10, 20, 30, 40]
        end

        @testset "in-place oneshot: Vector{Int} output 수용" begin
            out = zeros(Int, 4)
            constant_interp!(out, x_int, y_int, [0, 1, 2, 3])
            @test out == [10, 20, 30, 40]
        end
    end

    @testset "PeriodicBC" begin
        @testset "inclusive + Int Vector x + Int y" begin
            x = [0, 1, 2, 3]
            y = [10, 20, 30, 10]  # closed cycle
            itp = constant_interp(x, y; bc = PeriodicBC())
            @test itp isa ConstantInterpolant{Int, Int}
            @test itp(0.5) isa Int
            @test itp(3.5) isa Int  # wraps to first cell
        end

        @testset "exclusive + Int Vector x + Int y" begin
            x = [0, 1, 2]
            y = [10, 20, 30]
            itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 3))
            @test itp isa ConstantInterpolant{Int, Int}
            @test itp.x isa _ExclusivePeriodicAxis
            @test itp(0.5) isa Int
        end

        @testset "inclusive + Int Range x + Int y → _CachedRange{Int, Float64}" begin
            x = 0:1:3                # length 4 — closed cycle (x[end] == x[1] + period)
            y = [10, 20, 30, 10]     # closed: y[end] == y[1]
            itp = constant_interp(x, y; bc = PeriodicBC())
            @test itp isa ConstantInterpolant{Int, Int}
            @test itp.x isa _CachedRange{Int, Float64}
            @test itp(0.5) isa Int
            @test itp(0.5) == 10
        end
    end

    @testset "Series interp" begin
        @testset "ConstantSeriesInterpolant(Vector{Int}, Series(Vector{Int}, Vector{Int}))" begin
            x = [0, 1, 2, 3, 4]
            y1 = [10, 20, 30, 40, 50]
            y2 = [100, 200, 300, 400, 500]
            sitp = constant_interp(x, FastInterpolations.Series(y1, y2))
            @test sitp isa ConstantSeriesInterpolant
            r = sitp(1)  # scalar query — returns a tuple of K series values
            @test all(ri isa Int for ri in r)
            @test r[1] == 20
            @test r[2] == 200
        end

        @testset "scalar oneshot series: Int x + Series + Int xq → Int" begin
            x = [0, 1, 2, 3, 4]
            y1 = [10, 20, 30, 40, 50]
            s = FastInterpolations.Series(y1)
            r = constant_interp(x, s, 2)
            @test all(ri isa Int for ri in r)
            @test r[1] == 30
        end
    end

    @testset "Show output (non-Float eltype)" begin
        x = [0, 1, 2, 3]
        y = [10, 20, 30, 40]
        itp = constant_interp(x, y)
        s = repr("text/plain", itp)
        # Header should mention Int eltype (Int64 on 64-bit, Int32 on 32-bit).
        @test occursin("Int", s)
        @test occursin("ConstantInterpolant", s)
    end
end

# ============================================================================
# Group 2: Adjoint
# ============================================================================
@testitem "Constant eltype duck-type — adjoint" begin
    using LinearAlgebra: dot
    import FastInterpolations: ConstantAdjoint, _ConstantAnchoredQuery

    @testset "ConstantAdjoint with Int grid + Int xq — anchor type" begin
        x = collect(0:1:9)  # Vector{Int}
        xq = [2, 4, 6, 8]
        adj = constant_adjoint(x, xq)
        @test adj isa ConstantAdjoint
        @test eltype(adj.anchors) <: _ConstantAnchoredQuery{Int}
    end

    @testset "Dot-product identity (Int f + Int ȳ) under NearestSide" begin
        x = collect(0:1:9)
        xq = [2, 4, 6, 8]
        f = collect(10:10:100)  # Vector{Int}, length 10
        y_bar = [1, 2, 3, 4]    # Vector{Int}
        itp = constant_interp(x, f)
        adj = constant_adjoint(x, xq)
        # ⟨W·f, ȳ⟩ == ⟨f, Wᵀ·ȳ⟩  — exact equality with Int data
        @test dot(itp.(xq), y_bar) == dot(f, adj(y_bar))
    end

    @testset "Rational grid + Rational xq — Rational preserved" begin
        x = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1, 4 // 1]
        xq = Rational{Int}[1 // 2, 3 // 2]
        f = Rational{Int}[1 // 1, 2 // 1, 3 // 1, 4 // 1, 5 // 1]
        y_bar = Rational{Int}[1 // 1, 2 // 1]
        itp = constant_interp(x, f)
        adj = constant_adjoint(x, xq)
        # Rational arithmetic is exact — strict ==
        @test dot(itp.(xq), y_bar) == dot(f, adj(y_bar))
    end

    @testset "Side × extrap matrix (NoExtrap / ClampExtrap)" begin
        x = collect(0:1:9)
        xq = [2, 4, 6, 8]
        f = collect(10:10:100)
        y_bar = [1, 2, 3, 4]
        itp_args = (extrap = NoExtrap(),)
        for sd in (LeftSide(), RightSide(), NearestSide())
            for ex in (NoExtrap(), ClampExtrap())
                itp = constant_interp(x, f; side = sd, extrap = ex)
                adj = constant_adjoint(x, xq; side = sd, extrap = ex)
                @test dot(itp.(xq), y_bar) == dot(f, adj(y_bar))
            end
        end
    end
end

# ============================================================================
# Group 3: Float64 zero-alloc regression (no perf change on float path)
# ============================================================================
@testitem "Constant eltype duck-type — Float64 zero-alloc regression" setup = [AllocConstants] begin
    @testset "Scalar persistent eval @allocated unchanged" begin
        x = collect(0.0:0.1:1.0)
        y = sin.(x)
        itp = constant_interp(x, y)
        itp(0.5)  # warmup
        @test (@allocated itp(0.5)) <= ALLOC_THRESHOLD
    end

    @testset "Range Float64 scalar eval @allocated unchanged" begin
        x = 0.0:0.1:1.0
        y = collect(sin.(x))
        itp = constant_interp(x, y)
        itp(0.5)
        @test (@allocated itp(0.5)) <= ALLOC_THRESHOLD
    end

    @testset "Vector in-place loop @allocated unchanged" begin
        x = collect(0.0:0.1:1.0)
        y = sin.(x)
        xq = collect(0.05:0.1:0.95)
        out = similar(xq)
        itp = constant_interp(x, y)
        itp(out, xq)  # warmup
        @test (@allocated itp(out, xq)) <= ALLOC_THRESHOLD
    end
end
