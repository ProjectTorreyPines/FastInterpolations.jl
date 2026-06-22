# ═══════════════════════════════════════════════════════════════════════════════
# test_query_grid_promotion.jl
#
# Duck-type promotion contract for MISMATCHED query/grid element types.
#
# Oracle: a type-mismatched query (Int query on a Float grid, or vice versa) MUST
# return the same value as the naturally-promoted (all-Float) query — the package
# promotes, never coerces one side into the other's type. Covers value correctness,
# exact-boundary clamping, AD carrier, and type-stability of ND extrap handling
# (no OOB-vs-in-domain Union, which would leak to a public `Any` for Hermite ND).
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "Duck-type query/grid promotion across extraps (1D + ND)" begin
    approx(a, b) = isapprox(a, b; rtol = 1.0e-12, atol = 1.0e-12)

    # ── Grids ────────────────────────────────────────────────────────────
    # Fractional-endpoint Float grids so `Int(boundary)` would InexactError.
    xr_f = 0.5:1.0:9.5                 # Float64 range
    xv_f = collect(xr_f)               # Float64 vector
    # Int grids for the opposite direction (Float query must promote up).
    xr_i = 0:1:9                       # Int range
    xv_i = collect(xr_i)              # Int vector
    y1 = collect(Float64, 1:10)
    data = [Float64(i + j) for i in 1:10, j in 1:10]

    extraps = (ClampExtrap(), FillExtrap(fill_value = 0.0), ExtendExtrap(), WrapExtrap())

    # Method builders (each shares the ND coordinate-clamp path).
    build1d = (("linear", linear_interp), ("constant", constant_interp), ("cubic", cubic_interp))
    buildnd = build1d

    # ── 1D: Float grid + Int query (the bug direction; 1D was already green) ──
    @testset "1D Float grid + Int query" begin
        for (mname, mk) in build1d, ex in extraps, (gname, xg) in (("range", xr_f), ("vector", xv_f))
            exn = nameof(typeof(ex))
            @testset "$mname/$exn/$gname" begin
                itp = mk(xg, y1; extrap = ex)
                @test approx(itp(-5), itp(-5.0))   # OOB-left
                @test approx(itp(15), itp(15.0))   # OOB-right
                @test approx(itp(3), itp(3.0))     # in-bounds
            end
        end
    end

    # ── 1D: Int grid + Float query (opposite direction; lock against regression) ──
    @testset "1D Int grid + Float query" begin
        for (mname, mk) in build1d, ex in extraps, (gname, xg) in (("range", xr_i), ("vector", xv_i))
            exn = nameof(typeof(ex))
            @testset "$mname/$exn/$gname" begin
                itp = mk(xg, y1; extrap = ex)
                @test approx(itp(-2.5), itp(-2.5))  # OOB-left (trivially equal; checks no throw)
                @test approx(itp(11.5), itp(11.5))  # OOB-right
                @test approx(itp(3.5), itp(3.5))    # in-bounds
            end
        end
    end

    # ── ND (2D): Float grid + Int query (RED for ClampExtrap, all methods) ──
    @testset "ND Float grid + Int query" begin
        for (mname, mk) in buildnd, ex in extraps, (gname, g1, g2) in (("range", xr_f, xr_f), ("vector", xv_f, xv_f))
            exn = nameof(typeof(ex))
            @testset "$mname/$exn/$gname" begin
                itp = mk((g1, g2), data; extrap = ex)
                @test approx(itp(-5, 3), itp(-5.0, 3.0))    # OOB axis-1
                @test approx(itp(3, 15), itp(3.0, 15.0))    # OOB axis-2
                @test approx(itp(-5, 15), itp(-5.0, 15.0))  # OOB both
                @test approx(itp(3, 4), itp(3.0, 4.0))      # in-bounds
            end
        end
    end

    # ── ND (2D): Int grid + Float query (lock against regression) ──
    @testset "ND Int grid + Float query" begin
        for (mname, mk) in buildnd, ex in extraps, (gname, g1, g2) in (("range", xr_i, xr_i), ("vector", xv_i, xv_i))
            exn = nameof(typeof(ex))
            @testset "$mname/$exn/$gname" begin
                itp = mk((g1, g2), data; extrap = ex)
                @test approx(itp(-2.5, 3.0), itp(-2.5, 3.0))
                @test approx(itp(3.0, 11.5), itp(3.0, 11.5))
                @test approx(itp(3.5, 4.5), itp(3.5, 4.5))
            end
        end
    end

    # ── NoExtrap: in-bounds mismatched query must promote cleanly (no throw) ──
    # (OOB under NoExtrap is a DomainError by design — not exercised here.)
    @testset "NoExtrap in-bounds mismatched query" begin
        for (mname, mk) in build1d
            @testset "1D $mname Float grid + Int query" begin
                itp = mk(xr_f, y1; extrap = NoExtrap())
                @test approx(itp(3), itp(3.0))
            end
        end
        for (mname, mk) in buildnd
            @testset "ND $mname Float grid + Int query" begin
                itp = mk((xr_f, xr_f), data; extrap = NoExtrap())
                @test approx(itp(3, 4), itp(3.0, 4.0))
            end
        end
    end

    # ── DISCRIMINATING EXTREME CASES ─────────────────────────────────────
    # Pass under promotion; fail under weaker fixes: `oftype` (InexactError on
    # Int/fractional endpoint) or returning `q` unclamped (exact-boundary lock).
    # (clamp(q,lo,hi)'s sliver bug is pinned by test_fillextrap_domain_boundary.jl.)

    @testset "Exact boundary value — clamp lands on the grid corner node" begin
        # The OOB corner must clamp to the exact grid-corner datum, never an
        # extrapolation. RED under oftype: the Int OOB query throws before the
        # kernel runs. Also catches a fix that returns `q` unclamped.
        for (mname, mk) in buildnd, (gname, g1, g2) in (("range", xr_f, xr_f), ("vector", xv_f, xv_f))
            @testset "$mname/$gname" begin
                itp = mk((g1, g2), data; extrap = ClampExtrap())
                @test approx(itp(-5, -5), data[1, 1])      # both OOB-left  → corner (1,1)
                @test approx(itp(99, -5), data[end, 1])    # OOB-right x, OOB-left y
                @test approx(itp(-5, 99), data[1, end])    # OOB-left x, OOB-right y
                @test approx(itp(99, 99), data[end, end])  # both OOB-right → corner (end,end)
            end
        end
    end

    @testset "Non-half fractional grid endpoint — Int query stays InexactError-free" begin
        # The bug is not special to the 0.5 endpoint: ANY non-integer endpoint
        # makes oftype(Int, endpoint) throw. Pins the general case.
        xr2 = 0.3:1.0:9.3
        xv2 = collect(xr2)
        for (mname, mk) in buildnd, (gname, g1, g2) in (("range", xr2, xr2), ("vector", xv2, xv2))
            @testset "$mname/$gname" begin
                itp = mk((g1, g2), data; extrap = ClampExtrap())
                @test approx(itp(-5, 3), itp(-5.0, 3.0))
                @test approx(itp(15, 15), itp(15.0, 15.0))
            end
        end
    end
end

# ── AD robustness lock (ForwardDiff) ─────────────────────────────────────────
# Locks the clamp AD contract: Dual carrier preserved, zero gradient in the flat
# OOB region. Green today; guards against a weaker fix that would break it.
@testitem "ND ClampExtrap — AD carrier preserved, zero gradient in flat region" begin
    using ForwardDiff
    approx(a, b) = isapprox(a, b; rtol = 1.0e-12, atol = 1.0e-12)

    xr_f = 0.5:1.0:9.5
    data = [Float64(i + j) for i in 1:10, j in 1:10]

    for (mname, mk) in (("linear", linear_interp), ("constant", constant_interp), ("cubic", cubic_interp))
        @testset "$mname" begin
            itp = mk((xr_f, xr_f), data; extrap = ClampExtrap())

            # Gradient at OOB-left x, in-bounds y: the clamped axis is flat, so
            # ∂/∂x MUST be exactly 0; the in-bounds-axis partial stays finite.
            g = ForwardDiff.gradient(q -> itp(q[1], q[2]), [-5.0, 3.0])
            @test g[1] == 0.0
            @test isfinite(g[2])

            # Scalar Dual query stays a Dual (carrier preserved), value matches the
            # plain-Float clamp, partial w.r.t. the clamped axis is zero.
            r = itp(ForwardDiff.Dual{Nothing}(-5.0, 1.0), 3.0)
            @test r isa ForwardDiff.Dual
            @test approx(ForwardDiff.value(r), itp(-5.0, 3.0))
            @test ForwardDiff.partials(r)[1] == 0.0
        end
    end
end

# ── Type-stability of ND extrap handling (no OOB-vs-in-domain Union) ──────────
# A mismatched query eltype (Int/Float32 on a Float64 grid) must not make the OOB
# and in-domain branches differ in type — that Union costs union-split per query
# and leaks to a public `Any` for Hermite ND. Pinned internally + publicly.

@testitem "Type stability — ND _handle_all_extraps concrete for every (query-eltype, extrap)" begin
    using FastInterpolations: _handle_all_extraps
    # `_handle_all_extraps` promotes each axis query before dispatch → concrete
    # output tuple for every query eltype.
    gridsets = ((0.5:1.0:9.5, 0.5:1.0:9.5), (collect(0.5:1.0:9.5), collect(0.5:1.0:9.5)))
    exs = (
        NoExtrap(), ClampExtrap(), FillExtrap(fill_value = 0.0),
        ExtendExtrap(), WrapExtrap(), InBounds(),
    )
    for gs in gridsets, ex in exs, Q in (Int, Float32, Float64)
        extraps = (ex, ex)
        rt = Base.return_types(_handle_all_extraps, Tuple{Tuple{Q, Q}, typeof(gs), typeof(extraps)})
        @test length(rt) == 1 && isconcretetype(rt[1])
    end

    # Stronger: @inferred + isa pins the EXACT promoted coordinate type (Float64 grid
    # ⊕ any mismatched query eltype → Float64 coordinates). @inferred throws on a
    # non-concrete inference.
    g = (0.5:1.0:9.5, 0.5:1.0:9.5)
    for ex in (ClampExtrap(), FillExtrap(fill_value = 0.0), WrapExtrap(), ExtendExtrap(), InBounds())
        e2 = (ex, ex)
        @test (@inferred _handle_all_extraps((-5, -5), g, e2)) isa Tuple{Float64, Float64}        # Int OOB
        @test (@inferred _handle_all_extraps((3, 3), g, e2)) isa Tuple{Float64, Float64}          # Int in-domain
        @test (@inferred _handle_all_extraps((3.0f0, 3.0f0), g, e2)) isa Tuple{Float64, Float64}  # Float32
        @test (@inferred _handle_all_extraps((3.0, 3.0), g, e2)) isa Tuple{Float64, Float64}      # Float64
    end
end

@testitem "Type stability — ND public return concrete for mismatched query eltype (all methods)" begin
    g = 0.5:1.0:9.5
    data = [Float64(i + j) for i in 1:10, j in 1:10]
    builders = (
        linear_interp, constant_interp, cubic_interp, quadratic_interp,
        pchip_interp, cardinal_interp, akima_interp,
    )
    exs = (ClampExtrap(), FillExtrap(fill_value = 0.0), ExtendExtrap(), WrapExtrap())
    for mk in builders, ex in exs
        itp = mk((g, g), data; extrap = ex)
        for Q in (Int, Float32)
            rt = Base.return_types(itp, Tuple{Q, Q})
            @test length(rt) == 1 && isconcretetype(rt[1])
        end
    end

    # Stronger: @inferred + isa on the public call — pins the Hermite-family `Any`
    # leak (pchip/cardinal/akima) that union-split alone wouldn't expose.
    for mk in builders
        itp = mk((g, g), data; extrap = ClampExtrap())
        @test (@inferred itp(3, 4)) isa Float64      # in-domain Int query
        @test (@inferred itp(-5, 4)) isa Float64     # OOB Int query
    end
end

@testitem "Type stability — 1D public return concrete for mismatched query eltype (lock)" begin
    g = 0.5:1.0:9.5
    y = collect(Float64, 1:10)
    builders = (
        linear_interp, constant_interp, cubic_interp, quadratic_interp,
        pchip_interp, cardinal_interp, akima_interp,
    )
    exs = (ClampExtrap(), FillExtrap(fill_value = 0.0), ExtendExtrap(), WrapExtrap())
    for mk in builders, ex in exs, Q in (Int, Float32)
        itp = mk(g, y; extrap = ex)
        rt = Base.return_types(itp, Tuple{Q})
        @test length(rt) == 1 && isconcretetype(rt[1])
    end

    # Stronger: @inferred + isa (green locks — 1D value-clamps, so it is
    # structurally stable; this guards against a future 1D regression).
    for mk in builders
        itp = mk(g, y; extrap = ClampExtrap())
        @test (@inferred itp(3)) isa Float64       # in-domain Int query
        @test (@inferred itp(-5)) isa Float64      # OOB Int query
    end
end

# ── Type stability of the N=2 BATCH coordinate path (_locate_cell_2d_preamble) ──
# The persistent batch path (`_interp_nd_batch!`) hands the RAW per-query tuple straight
# to the N=2 `_locate_cell` specialization → `_locate_cell_2d_preamble`, which — unlike
# the generic-N `_handle_all_extraps` chokepoint — did NOT promote. So a mismatched-eltype
# query (Int on a Float64 grid) made the ClampExtrap OOB branch return Float64 while the
# in-domain branch returned the Int `q` → a `Union{Int,Float64}` `x_eval`/`y_eval` that
# union-splits per query inside the batch loop. (The scalar path promotes up in
# `_eval_nd_at_point`, so the Union is hidden there; the leak is batch-only, and it does
# not escape `_locate_cell`'s own return type — only the preamble exposes it.)
@testitem "Type stability — N=2 _locate_cell_2d_preamble concrete for mismatched query eltype" begin
    using FastInterpolations: _locate_cell_2d_preamble

    g = 0.5:1.0:9.5
    data = [Float64(i + j) for i in 1:10, j in 1:10]
    itp = linear_interp((g, g), data; extrap = ClampExtrap())

    grids = itp.grids
    extraps = itp.extraps
    policies = itp.searches          # (AutoSearch(), AutoSearch())
    hints = (Ref(1), Ref(1))
    mono = (false, false)

    # Float64 grid ⊕ Int query → Float64 coordinates → fully concrete 6-tuple.
    RT = Tuple{Float64, Float64, Int, Int, Float64, Float64}
    @test (@inferred _locate_cell_2d_preamble((3, 4), grids, extraps, policies, hints, mono)) isa RT     # both in-domain
    @test (@inferred _locate_cell_2d_preamble((-5, 4), grids, extraps, policies, hints, mono)) isa RT    # OOB axis-1
    @test (@inferred _locate_cell_2d_preamble((3, 99), grids, extraps, policies, hints, mono)) isa RT    # OOB axis-2
end
