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
        @testset "Int Vector x, Int y — output follows query type" begin
            x = [0, 1, 2, 3, 4]
            y = [10, 20, 30, 40, 50]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Int, Int}
            @test eltype(itp.y) === Int
            @test itp.x isa _CachedVector{Int, Float64}
            # Float xq → Float (Int y * one(Float) = Float).
            @test itp(1.5) isa Float64
            @test itp(1.5) == 20.0
            # Int xq → Int (fully-Int chain preserves Tv).
            @test itp(2) isa Int
            @test itp(2) == 30
        end

        @testset "Int Range x, Int y → itp.x::_CachedRange{Int, Float64}" begin
            x = 0:1:4
            y = [10, 20, 30, 40, 50]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Int, Int}
            @test itp.x isa _CachedRange{Int, Float64}
            @test itp.x.h === 1
            @test itp.x.inv_h === 1.0
            # Float xq → Float.
            @test itp(2.5) isa Float64
            @test itp(2.5) == 30.0
            # Int xq → Int.
            @test itp(3) isa Int
            @test itp(3) == 40
        end

        @testset "Rational x, Rational y, Rational xq → output Rational{Int}" begin
            x = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1, 4 // 1]
            y = Rational{Int}[1 // 2, 3 // 2, 5 // 2, 7 // 2, 9 // 2]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Rational{Int}, Rational{Int}}
            # Rational xq stays Rational (Rational * one(Rational) = Rational).
            @test itp(3 // 2) isa Rational{Int}
            @test itp(3 // 2) === 3 // 2
        end

        @testset "Float64 x, Int y, Float xq → Float (natural promote)" begin
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = [10, 20, 30, 40, 50]
            itp = constant_interp(x, y)
            @test itp isa ConstantInterpolant{Float64, Int}
            # Carrier propagates: Int y * one(Float64) = Float64. Value preserved.
            @test itp(1.5) isa Float64
            @test itp(1.5) == 20.0
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

        @testset "vector alloc oneshot returns Vector{Int}" begin
            v = constant_interp(x_int, y_int, [0, 1, 2, 3])
            @test v isa Vector{Int}
            @test v == [10, 20, 30, 40]
        end

        @testset "in-place oneshot accepts Vector{Int} output" begin
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
            # Float xq → Float (natural promote); Int xq → Int.
            @test itp(0.5) isa Float64
            @test itp(3.5) isa Float64    # wraps to first cell
            @test itp(1) isa Int
        end

        @testset "exclusive + Int Vector x + Int y" begin
            # Constant grid eltype stays Int (`:exclusive` extension is
            # shape-only, n → n+1). Query type drives output.
            x = [0, 1, 2]
            y = [10, 20, 30]
            itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 3))
            @test itp isa ConstantInterpolant{Int, Int}
            @test length(itp.x) == 4          # closed-cycle n+1
            @test itp.x[end] == itp.x[1] + 3  # virtual endpoint at period
            @test itp(0.5) isa Float64
            @test itp(1) isa Int
        end

        @testset "inclusive + Int Range x + Int y → _CachedRange{Int, Float64}" begin
            x = 0:1:3                # length 4 — closed cycle (x[end] == x[1] + period)
            y = [10, 20, 30, 10]     # closed: y[end] == y[1]
            itp = constant_interp(x, y; bc = PeriodicBC())
            @test itp isa ConstantInterpolant{Int, Int}
            @test itp.x isa _CachedRange{Int, Float64}
            @test itp(0.5) isa Float64
            @test itp(0.5) == 10.0
            @test itp(1) isa Int
            @test itp(1) == 20
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

# ============================================================================
# Group 4: ND — selection kernel generalizes to per-axis cell pick
# ============================================================================
# Constant ND follows the same raw-eltype contract:
#   Tg = promote_type(eltype.(grids)...)   (no `float()` widening)
#   Tv = eltype(data)
# Container heterogeneity (Range × Vector axes) was always supported via
# `_convert_grids_typed`; eltype heterogeneity is unified to a single Tg via
# `promote_type` (Int × Float → Float, Int × Rational → Rational, …).
@testitem "Constant eltype duck-type — ND forward + adjoint" begin
    using LinearAlgebra: dot
    import FastInterpolations: ConstantInterpolantND, _CachedRange, _CachedVector

    @testset "2D persistent — homogeneous eltypes" begin
        @testset "Int × Int axes, Int data → ConstantInterpolantND{Int, Int, 2}" begin
            x = 0:1:4
            y = [0, 1, 2, 3]
            data = [10 * i + j for i in 1:5, j in 1:4]   # Int matrix
            itp = constant_interp((x, y), data)
            @test itp isa ConstantInterpolantND{Int, Int, 2}
            # Float xq → Float (natural promote); Int xq → Int (fully-Int chain).
            @test itp((2.5, 1.5)) isa Float64
            @test itp((2, 1)) isa Int
            @test itp.grids[1] isa _CachedRange{Int, Float64}
            @test itp.grids[2] isa _CachedVector{Int, Float64}
        end

        @testset "Rational × Rational axes, Rational data" begin
            x = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1]
            y = Rational{Int}[0 // 1, 1 // 1, 2 // 1]
            data = Rational{Int}[i // 1 + j // 2 for i in 1:4, j in 1:3]
            itp = constant_interp((x, y), data)
            @test itp isa ConstantInterpolantND{Rational{Int}, Rational{Int}, 2}
            @test itp((3 // 2, 1 // 2)) isa Rational{Int}
        end

        @testset "Float32 × Float32 axes → Float32 preserved (regression guard)" begin
            x = Float32[0, 1, 2, 3, 4]
            y = Float32[0, 1, 2, 3]
            data = Float32[i + j for i in 1:5, j in 1:4]
            itp = constant_interp((x, y), data)
            @test itp isa ConstantInterpolantND{Float32, Float32, 2}
            @test itp((2.5f0, 1.5f0)) isa Float32
        end
    end

    @testset "2D persistent — heterogeneous container types (Range × Vector)" begin
        # Container heterogeneity was always supported; pin it under the new
        # raw-eltype policy too. Both axes share a single Tg via promote_type.
        x = 0:1:4                # Int Range
        y = [0.0, 1.0, 2.0, 3.0] # Float Vector
        data = [Float64(i + j) for i in 1:5, j in 1:4]
        itp = constant_interp((x, y), data)
        # promote_type(Int, Float64) == Float64 → Tg unified.
        @test itp isa ConstantInterpolantND{Float64, Float64, 2}
        @test itp((2.5, 1.5)) isa Float64
    end

    @testset "Int data + Float axes — query carrier drives output" begin
        # Axes Float, data Int → Tv=Int, Tg=Float64.
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 2.0]
        data = [10 * i + j for i in 1:4, j in 1:3]  # Int matrix
        itp = constant_interp((x, y), data)
        @test itp isa ConstantInterpolantND{Float64, Int, 2}
        # Float xq → Float result (Int data * one(Float) = Float).
        @test itp((1.5, 0.5)) isa Float64
    end

    @testset "3D Int^3 — fully-Int chain stays Int" begin
        x = 0:1:3
        y = 0:1:2
        z = 0:1:2
        data = [i + 10j + 100k for i in 1:4, j in 1:3, k in 1:3]  # Int 3D
        itp = constant_interp((x, y, z), data)
        @test itp isa ConstantInterpolantND{Int, Int, 3}
        # Int xq → Int (fully-Int chain).
        @test itp((1, 0, 1)) isa Int
        # Float xq → Float.
        @test itp((1.5, 0.5, 1.5)) isa Float64
    end

    @testset "PeriodicBC + Int axes (inclusive)" begin
        x = 0:1:3                # length 4, closed cycle in y data
        y = 0:1:2
        # Closed-cycle data: data[end, :] == data[1, :] (period 3 in x).
        data = [(i == 4 ? 1 : i) + 10j for i in 1:4, j in 1:3]
        itp = constant_interp((x, y), data; bc = (PeriodicBC(), NoBC()))
        @test itp isa ConstantInterpolantND{Int, Int, 2}
        @test itp((0.5, 1.5)) isa Float64
        @test itp((1, 1)) isa Int
    end

    @testset "ND oneshot scalar — Float xq → Float (natural promote)" begin
        x = 0:1:4
        y = 0:1:3
        data = [10 * i + j for i in 1:5, j in 1:4]
        # Float xq carrier propagates: Int data * one(Float64) = Float64.
        r = constant_interp((x, y), data, (2.5, 1.5))
        @test r isa Float64
    end

    @testset "ND oneshot batch — current behavior pinned (allocator follow-up)" begin
        x = 0:1:4
        y = 0:1:3
        data = [10 * i + j for i in 1:5, j in 1:4]
        queries = [(2.5, 1.5), (0.5, 0.5), (3.5, 2.5)]
        # ND oneshot 3-arg batch allocator does not yet sample-first; pin
        # current Vector{Int} return so a future migration is intentional.
        vals = constant_interp((x, y), data, queries)
        @test vals isa Vector{Int}
        @test length(vals) == 3
    end

    @testset "ND adjoint — Int grid + Int xq dot-product identity (exact ==)" begin
        x = collect(0:1:5)
        y = collect(0:1:4)
        f = [10 * i + j for i in 1:6, j in 1:5]    # Int data
        queries = [(2, 1), (4, 3), (1, 2)]
        y_bar = [1, 2, 3]
        itp = constant_interp((x, y), f)
        adj = constant_adjoint((x, y), queries)
        # Exact equality with Int — selection kernel + Int data + Int weights.
        @test dot([itp(q) for q in queries], y_bar) == dot(vec(f), vec(adj(y_bar)))
    end
end

@testitem "Constant eltype duck-type — ND Float64 zero-alloc regression" setup = [AllocConstants] begin
    @testset "2D scalar eval @allocated unchanged (Float64 path)" begin
        x = collect(0.0:0.1:1.0)
        y = collect(0.0:0.1:1.0)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        itp = constant_interp((x, y), data)
        itp((0.5, 0.5))   # warmup
        @test (@allocated itp((0.5, 0.5))) <= ND_ALLOC_THRESHOLD
    end
end

# ============================================================================
# Group 5: ComplexF32 raw-eltype (no silent widening to ComplexF64)
# ============================================================================
@testitem "Constant eltype duck-type — ComplexF32 stays ComplexF32" begin
    using LinearAlgebra: dot
    import FastInterpolations: ConstantInterpolant, ConstantInterpolantND

    @testset "1D persistent" begin
        x = Float32.(0:4)
        y = ComplexF32[1 + 1im, 2 + 2im, 3 + 3im, 4 + 4im, 5 + 5im]
        itp = constant_interp(x, y)
        @test itp isa ConstantInterpolant{Float32, ComplexF32}
        @test itp(1.5f0) === ComplexF32(2 + 2im)
        @test eltype(itp.(Float32[0.5, 1.5, 2.5])) === ComplexF32
    end

    @testset "1D series" begin
        x = Float32.(0:4)
        y1 = ComplexF32[1, 2, 3, 4, 5]
        y2 = ComplexF32[10, 20, 30, 40, 50]
        sitp = constant_interp(x, Series(y1, y2))
        out = sitp(1.5f0)
        @test eltype(out) === ComplexF32
        @test out == ComplexF32[2, 20]
    end

    @testset "ND persistent" begin
        x = Float32.(0:3); y = Float32.(0:3)
        data = ComplexF32[(i + j) + (i - j)im for i in 1:4, j in 1:4]
        itp = constant_interp((x, y), data)
        @test itp isa ConstantInterpolantND{Float32, ComplexF32, 2}
        @test itp((1.5f0, 2.5f0)) === data[2, 3]
    end

    @testset "ND adjoint (Float32 grid, ComplexF32 y_bar)" begin
        x = Float32.(0:3); y = Float32.(0:3)
        data = ComplexF32[(i + j) + (i - j)im for i in 1:4, j in 1:4]
        itp = constant_interp((x, y), data)
        queries = [(0.5f0, 0.5f0), (1.5f0, 2.5f0), (2.5f0, 0.5f0)]
        y_bar = ComplexF32[1 + 1im, 2 - 1im, 3 + 0im]
        adj = constant_adjoint((x, y), queries)
        out = adj(y_bar)
        @test eltype(out) === ComplexF32
        @test dot(itp.(queries), y_bar) ≈ dot(data, out)
    end
end

# ============================================================================
# Group 6: FillExtrap × duck-typed Y (raw-Tv fill value contract)
# ============================================================================
@testitem "Constant eltype duck-type — FillExtrap raw-Tv contract" begin
    @testset "1D persistent: Int data + Int fill — natural promote with Float xq" begin
        x = Float64.(0:4)
        y = [10, 20, 30, 40, 50]
        itp = constant_interp(x, y; extrap = FillExtrap(-1))
        # Construction promotes the fill value to Tv (Int) — raw storage.
        @test itp.extrap === FillExtrap{Int}(-1)
        # Float xq → Float result for both in-domain and OOB (carrier propagates).
        @test itp(1.5) === 20.0
        @test itp(-1.0) === -1.0
    end

    @testset "1D persistent: Int data + Float fill → InexactError on construction" begin
        x = Float64.(0:4)
        y = [10, 20, 30, 40, 50]
        @test_throws InexactError constant_interp(x, y; extrap = FillExtrap(NaN))
    end

    @testset "ND persistent: ComplexF64 data + Complex fill → Complex output" begin
        x = collect(0.0:1:3); y = collect(0.0:1:3)
        data = ComplexF64[(i + j) + (i - j)im for i in 1:4, j in 1:4]
        fill_val = ComplexF64(-1 + 0im)
        itp = constant_interp((x, y), data; extrap = FillExtrap(fill_val))
        @test itp((-1.0, -1.0)) === fill_val
        @test eltype(itp.([(0.5, 0.5), (-1.0, 5.0)])) === ComplexF64
    end
end

# ============================================================================
# Group 7: @inferred coverage on the new branched output-type paths
# ============================================================================
@testitem "Constant eltype duck-type — @inferred coverage" begin
    import FastInterpolations: ConstantInterpolantND

    @testset "1D Series scalar — Tv pass-through branch" begin
        x = collect(0.0:0.1:1.0)
        y1 = collect(1:11)
        y2 = collect(11:21)
        sitp = constant_interp(x, Series(y1, y2))
        @test @inferred(sitp(0.55)) isa Vector{Int}
    end

    @testset "1D Series scalar — ComplexF32 Tv branch" begin
        x = Float32.(0:0.1:1.0)
        y1 = ComplexF32.(1:11)
        sitp = constant_interp(x, Series(y1))
        @test @inferred(sitp(0.55f0)) isa Vector{ComplexF32}
    end

    @testset "ND persistent forward (Int data, Float xq → Float carrier)" begin
        x = collect(0.0:1:3); y = collect(0.0:1:3)
        data = [10 * i + j for i in 1:4, j in 1:4]
        itp = constant_interp((x, y), data)
        @test itp isa ConstantInterpolantND{Float64, Int, 2}
        @test @inferred(itp((1.5, 2.5))) === Float64(data[2, 3])
    end
end

# ============================================================================
# Group 8: Anchor query precision — Tq = promote_type(Tg, eltype(xq))
# ============================================================================
# Narrower grids (Int, Float32) must not truncate wider queries — verified
# end-to-end via the forward/adjoint dot-product identity.
@testitem "Constant anchor: Tq = promote_type(Tg, eltype(xq))" begin
    using LinearAlgebra: dot
    import FastInterpolations: ConstantAdjoint, _ConstantAnchoredQuery

    @testset "Int grid + Float query — 1D adjoint, dot identity" begin
        x = collect(0:9)
        y = collect(10.0:10.0:100.0)
        xq = [0.5, 1.5, 2.5, 3.5]
        y_bar = [0.1, 0.2, -0.3, 0.4]

        itp = constant_interp(x, y)
        adj = constant_adjoint(x, xq)
        @test adj isa ConstantAdjoint{Int, Float64}
        @test eltype(adj.anchors) === _ConstantAnchoredQuery{Int, Float64}
        @test dot(itp.(xq), y_bar) ≈ dot(y, adj(y_bar))
    end

    @testset "Int grid + Float query — Series vector eval" begin
        x = collect(0:4)
        y1 = collect(10.0:10.0:50.0)
        y2 = collect(100.0:100.0:500.0)
        sitp = constant_interp(x, Series(y1, y2))
        result = sitp([0.5, 1.5, 2.5])  # fractional Float queries on Int grid
        @test result == [[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]
    end

    @testset "Int grids + Float queries — ND adjoint, dot identity" begin
        x = collect(0:3); y = collect(0:3)
        data = Float64[10 * i + j for i in 1:4, j in 1:4]
        queries = [(0.5, 1.5), (1.5, 2.5), (2.5, 0.5)]
        y_bar = [1.0, -0.5, 0.7]

        itp = constant_interp((x, y), data)
        adj = constant_adjoint((x, y), queries)
        @test dot(itp.(queries), y_bar) ≈ dot(data, adj(y_bar))
    end

    @testset "Float32 grid + Float64 query — NearestSide tie-break consistency" begin
        x = Float32[0, 1, 2]
        y = Float32[10, 20, 30]
        xq = [nextfloat(0.5)]   # strictly > 0.5 in Float64; collapses to 0.5f0 if narrowed
        y_bar = [1.0]

        itp = constant_interp(x, y)
        adj = constant_adjoint(x, xq)
        # Carrier of Float64 xq propagates into result: Float32 y * one(Float64) = Float64.
        @test itp(xq[1]) === 20.0
        @test adj(y_bar) ≈ Float32[0, 1, 0]
        @test dot([itp(xq[1])], y_bar) ≈ dot(y, adj(y_bar))
    end

    @testset "Float32 ND grids + Float64 queries — tie-break consistency" begin
        x = Float32[0, 1, 2]
        y = Float32[0, 1, 2]
        data = Float64[10 * i + j for i in 1:3, j in 1:3]
        queries = [(nextfloat(0.5), nextfloat(0.5))]
        y_bar = [1.0]

        itp = constant_interp((x, y), data)
        adj = constant_adjoint((x, y), queries)
        @test dot([itp(queries[1])], y_bar) ≈ dot(data, adj(y_bar))
    end
end

# ============================================================================
# Group 9: Oneshot duck-typed query passthrough (Dual carrier preserved)
# ============================================================================
# Persistent callables already widen `T_out = promote_type(Tv, Tq)` for non-
# `_PromotableValue` queries. The allocating oneshot paths must match — else
# AD callers silently get raw Tv back, stripping their Dual carrier.
@testitem "Constant oneshot: duck queries widen, plain queries stay raw" begin
    using ForwardDiff
    using ForwardDiff: Dual

    x = [0.0, 1.0, 2.0, 3.0]
    y = [10, 20, 30, 40]
    s = Series(y)
    xq_d = Dual(0.5, 1.0)

    @testset "vector oneshot: Dual xq → Dual output" begin
        out = constant_interp(x, y, [xq_d, Dual(1.5, 1.0)])
        @test eltype(out) <: Dual
        @test ForwardDiff.value.(out) == [10.0, 20.0]
    end

    @testset "scalar series oneshot: Dual xq → Dual output" begin
        out = constant_interp(x, s, xq_d)
        @test eltype(out) <: Dual
        @test ForwardDiff.value.(out) == [10.0]
    end

    @testset "batch series oneshot: Dual xq → Dual output" begin
        out = constant_interp(x, s, [xq_d])
        @test eltype(eltype(out)) <: Dual
        @test ForwardDiff.value.(out[1]) == [10.0]
    end

    @testset "plain Float xq carrier propagation — plain vs Series oneshot" begin
        # Plain-vector oneshot picks up the Float64 xq carrier (Tv=Int → Float).
        @test constant_interp(x, y, [0.5, 1.5]) isa Vector{Float64}
        # Series oneshot allocator does not yet sample-first; pin current
        # Vector{Int} return so a future migration is intentional.
        @test constant_interp(x, s, 0.5) isa Vector{Int}
        @test constant_interp(x, s, [0.5, 1.5]) isa Vector{Vector{Int}}
    end
end

# ============================================================================
# Group 10: Natural promote — output eltype = `_output_eltype(Tv, Tg, Tq)`
# ============================================================================
# Constant's kernel propagates `Tq` via `* one(dL)`, so scalar / `itp([xq])` /
# `itp.([xq])` / oneshot all agree on the naturally-promoted type. `Int y` +
# `Float xq` → `Float`; `Int y` + `Int xq` → `Int`; `Int y` + `Dual xq` →
# `Dual` (carrier preserved without a Constant-specific code path).
@testitem "Constant persistent: scalar + batch follow natural promote" begin
    using ForwardDiff
    using ForwardDiff: Dual

    @testset "1D Int y + Float xq → Float" begin
        itp = constant_interp([0.0, 1.0, 2.0, 3.0], [10, 20, 30, 40])
        @test itp(0.5) === 10.0
        @test itp([0.5]) == [10.0]
        @test itp([0.5]) isa Vector{Float64}
        @test itp.([0.5]) isa Vector{Float64}
        @test constant_interp([0.0, 1.0, 2.0, 3.0], [10, 20, 30, 40], [0.5]) isa Vector{Float64}
    end

    @testset "1D Int y + Int xq → Int (fully-Int chain preserves Tv)" begin
        itp = constant_interp([0, 1, 2, 3], [10, 20, 30, 40])
        @test itp(0) === 10
        @test itp([0, 1]) isa Vector{Int}
        @test constant_interp([0, 1, 2, 3], [10, 20, 30, 40], [0, 1]) isa Vector{Int}
    end

    @testset "ND Int data + Float xq → Float (persistent paths)" begin
        x = [0.0, 1.0]; y = [0.0, 1.0]; data = [10 20; 30 40]
        itp = constant_interp((x, y), data)
        @test itp((0.5, 0.5)) === 10.0
        @test itp([(0.5, 0.5)]) == [10.0]
        @test itp([(0.5, 0.5)]) isa Vector{Float64}
        # ND oneshot 3-arg allocator pending sample-first migration; pin current.
        @test constant_interp((x, y), data, [(0.5, 0.5)]) isa Vector{Int}
    end

    @testset "1D persistent Dual → carrier preserved (scalar + batch)" begin
        itp = constant_interp([0.0, 1.0, 2.0, 3.0], [10, 20, 30, 40])
        d = Dual(0.5, 1.0)
        @test itp(d) isa Dual{Nothing, Float64, 1}
        @test itp([d]) isa Vector{<:Dual{Nothing, Float64, 1}}
        @test itp.([d]) isa Vector{<:Dual{Nothing, Float64, 1}}
    end

    @testset "ND persistent Dual → carrier preserved (scalar + batch)" begin
        x = [0.0, 1.0]; y = [0.0, 1.0]; data = [10 20; 30 40]
        itp = constant_interp((x, y), data)
        d = Dual(0.5, 1.0)
        @test itp((d, d)) isa Dual{Nothing, Float64, 1}
        @test itp([(d, d)]) isa Vector{<:Dual{Nothing, Float64, 1}}
    end
end
