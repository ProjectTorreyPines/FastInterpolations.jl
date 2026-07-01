# Behavior pin for the canonical `_UnitStep` + `InBounds` ND locate fast path.
#
# This is a perf refactor (the correct results already exist via the generic
# `_locate_cell`), so these are characterization tests: green on the pre-refactor
# baseline and green after. Two assertions carry real signal for the refactor:
#   1. value/deriv match an INDEPENDENT manual multilinear reference (catches any
#      idx/α/stencil mistake in the fast path), and
#   2. the InBounds path is BIT-IDENTICAL (`===`) to the NoExtrap path on the same
#      `_UnitStep` grid for in-domain queries — this fails the moment the fast
#      path's arithmetic deviates by even one ULP from the generic path.

@testitem "UnitStep InBounds locate — 2D matches reference, bit-identical to NoExtrap" begin
    using FastInterpolations: InBounds, DerivOp, GridIdx

    # Independent bilinear reference on unit-step axes (node index == position).
    function ref_bilin(axx, axy, data, qx, qy)
        lox, loy = first(axx), first(axy)
        nx, ny = length(axx), length(axy)
        ix = clamp(floor(Int, qx - lox) + 1, 1, nx - 1); αx = qx - (lox + (ix - 1))
        iy = clamp(floor(Int, qy - loy) + 1, 1, ny - 1); αy = qy - (loy + (iy - 1))
        v0 = (1 - αx) * data[ix, iy] + αx * data[ix + 1, iy]
        v1 = (1 - αx) * data[ix, iy + 1] + αx * data[ix + 1, iy + 1]
        return (1 - αy) * v0 + αy * v1
    end

    axx, axy = 1:7, 2:9                      # two different 1-step lo's (1 and 2)
    data = [sin(0.3i) + cos(0.2j) + 0.01i * j for i in 1:length(axx), j in 1:length(axy)]
    inb = linear_interp((axx, axy), data; extrap = (InBounds(), InBounds()))
    nox = linear_interp((axx, axy), data)    # default NoExtrap, same _UnitStep grids

    qs = ((1.0, 2.0), (1.25, 4.75), (3.4, 6.2), (6.999, 8.5), (7.0, 9.0))  # incl. q==hi
    @testset "value: matches reference and === NoExtrap" begin
        for q in qs
            @test inb(q) ≈ ref_bilin(axx, axy, data, q...)
            @test inb(q) === nox(q)          # bit-identical to the generic path
        end
    end

    @testset "right-boundary q==hi returns the top corner" begin
        @test inb((7.0, 9.0)) === data[end, end]
    end

    @testset "first derivatives match reference slopes (deriv stays generic)" begin
        for q in ((1.25, 4.75), (3.4, 6.2))
            qx, qy = q
            ix = clamp(floor(Int, qx - 1) + 1, 1, length(axx) - 1)
            iy = clamp(floor(Int, qy - 2) + 1, 1, length(axy) - 1)
            αx = qx - ix; αy = qy - (1 + iy)         # node positions: x→ix, y→iy+1
            dvdx = (1 - αy) * (data[ix + 1, iy] - data[ix, iy]) +
                αy * (data[ix + 1, iy + 1] - data[ix, iy + 1])
            dvdy = (1 - αx) * (data[ix, iy + 1] - data[ix, iy]) +
                αx * (data[ix + 1, iy + 1] - data[ix + 1, iy])
            @test inb(q; deriv = DerivOp(1, 0)) ≈ dvdx
            @test inb(q; deriv = DerivOp(0, 1)) ≈ dvdy
            @test inb(q; deriv = DerivOp(1, 0)) === nox(q; deriv = DerivOp(1, 0))
            @test inb(q; deriv = DerivOp(0, 1)) === nox(q; deriv = DerivOp(0, 1))
        end
    end

    @testset "GridIdx queries === NoExtrap" begin
        @test inb((GridIdx(7), 8.5)) === nox((GridIdx(7), 8.5))
        @test inb((3.25, GridIdx(8))) === nox((3.25, GridIdx(8)))
        @test inb((GridIdx(1), GridIdx(1))) === data[1, 1]
    end

    @testset "looped scalar eval is non-allocating" begin
        # Loop barrier matches real looped-scalar use (imresize); the per-call
        # kwarg overhead a single `@allocated` would catch is elided in the loop.
        function fill_loop!(out, itp, qxs, qys)
            @inbounds for i in eachindex(qxs, qys)
                out[i] = itp(qxs[i], qys[i])
            end
            return out
        end
        qxs = collect(range(1.05, 6.95; length = 64))
        qys = collect(range(2.05, 8.95; length = 64))
        out = similar(qxs)
        fill_loop!(out, inb, qxs, qys)               # warmup/compile
        @test (@allocated fill_loop!(out, inb, qxs, qys)) == 0
    end
end

@testitem "UnitStep InBounds locate — 3D matches reference, bit-identical to NoExtrap" begin
    using FastInterpolations: InBounds

    function ref_trilin(ax, data, qx, qy, qz)
        lo = map(first, ax); n = map(length, ax)
        ix = clamp(floor(Int, qx - lo[1]) + 1, 1, n[1] - 1); αx = qx - (lo[1] + (ix - 1))
        iy = clamp(floor(Int, qy - lo[2]) + 1, 1, n[2] - 1); αy = qy - (lo[2] + (iy - 1))
        iz = clamp(floor(Int, qz - lo[3]) + 1, 1, n[3] - 1); αz = qz - (lo[3] + (iz - 1))
        c(a, b, t) = (1 - t) * a + t * b
        v00 = c(data[ix, iy, iz], data[ix + 1, iy, iz], αx)
        v10 = c(data[ix, iy + 1, iz], data[ix + 1, iy + 1, iz], αx)
        v01 = c(data[ix, iy, iz + 1], data[ix + 1, iy, iz + 1], αx)
        v11 = c(data[ix, iy + 1, iz + 1], data[ix + 1, iy + 1, iz + 1], αx)
        return c(c(v00, v10, αy), c(v01, v11, αy), αz)
    end

    ax = (1:5, 1:6, 1:7)
    data = [0.1i + 0.2j + 0.3k + 0.01i * j * k for i in 1:5, j in 1:6, k in 1:7]
    inb = linear_interp(ax, data; extrap = (InBounds(), InBounds(), InBounds()))
    nox = linear_interp(ax, data)

    for q in ((1.0, 1.0, 1.0), (2.3, 4.1, 5.9), (4.999, 5.5, 6.001), (5.0, 6.0, 7.0))
        @test inb(q) ≈ ref_trilin(ax, data, q...)
        @test inb(q) === nox(q)
    end
end

@testitem "per-axis extrap: lean search applies per axis (mixed InBounds/Clamp)" begin
    # The lean InBounds search is threaded per-axis through `_search_axis_adaptive`, so a
    # mixed `(InBounds, ClampExtrap)` interpolant leans ONLY the InBounds axis while the
    # Clamp axis keeps its domain handling. In-domain queries must stay bit-identical.
    using FastInterpolations: InBounds, ClampExtrap

    data = [0.1i + 0.3j + 0.01i * j for i in 1:7, j in 1:8]
    grids = (1:7, 2:9)
    mixed = linear_interp(grids, data; extrap = (InBounds(), ClampExtrap()))
    allclamp = linear_interp(grids, data; extrap = (ClampExtrap(), ClampExtrap()))
    nox = linear_interp(grids, data)

    @testset "in-domain: InBounds (lean) axis === domain-checked paths" begin
        for q in ((1.5, 3.5), (4.25, 6.75), (7.0, 9.0))
            @test mixed(q) === nox(q)
            @test mixed(q) === allclamp(q)
        end
    end

    @testset "Clamp axis still clamps while InBounds axis stays in-domain" begin
        # y above the grid → Clamp axis pins to last; x (InBounds) in-domain, no check.
        @test mixed((3.5, 100.0)) === allclamp((3.5, 100.0))
    end
end

@testitem "1D InBounds lean vs ExtendExtrap OOB — lean only the genuinely-in-domain path" begin
    # RED pin for the 1D lean search. The lean `_search_direct_inbounds` uses a ONE-SIDED
    # clamp (no lower `max(·,1)`), valid ONLY when the query is in-domain. `ExtendExtrap`
    # reaches the eval core with an OOB query and extrapolates off the *boundary cell* — it
    # MUST take the standard two-sided-clamp search, not the lean one. If the generic
    # `::AbstractExtrap` eval delegates an OOB ExtendExtrap query to the lean core, the
    # one-sided clamp returns idx ≤ 0 → BoundsError. This test fails (errors) on that bug
    # and passes once InBounds (lean) and ExtendExtrap (clamp) are split.
    using FastInterpolations: InBounds, ExtendExtrap

    x = 1.0:1.0:12.0
    y = [sin(0.3i) + 0.1i for i in 1:length(x)]
    qs_in = (1.5, 3.4, 6.25, 9.99, 12.0)         # in-domain
    q_left = 0.4                                  # OOB-left  (< x[1] = 1)
    q_right = 13.7                                # OOB-right (> x[end] = 12)

    @testset "genuine InBounds === NoExtrap in-domain (lean is bit-identical)" begin
        for f in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            itp_ib = f(x, y; extrap = InBounds())
            itp_ne = f(x, y)
            for q in qs_in
                @test itp_ib(q) === itp_ne(q)
            end
        end
    end

    @testset "ExtendExtrap OOB extrapolates (must NOT hit the lean one-sided clamp)" begin
        for f in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            itp_ext = f(x, y; extrap = ExtendExtrap())
            @test isfinite(itp_ext(q_left))      # OOB-left: lean would BoundsError at idx ≤ 0
            @test isfinite(itp_ext(q_right))     # OOB-right
        end
    end

    @testset "linear ExtendExtrap matches the boundary-segment extension (value pin)" begin
        li = linear_interp(x, y; extrap = ExtendExtrap())
        @test li(q_left) ≈ y[1] + (q_left - x[1]) * (y[2] - y[1]) / (x[2] - x[1])
        @test li(q_right) ≈ y[end - 1] + (q_right - x[end - 1]) * (y[end] - y[end - 1]) / (x[end] - x[end - 1])
    end

    @testset "hermite (precomputed slopes): InBounds lean + ExtendExtrap OOB safe" begin
        dy = [0.3cos(0.3i) + 0.1 for i in 1:length(x)]
        hib = hermite_interp(x, y, dy; extrap = InBounds())
        hne = hermite_interp(x, y, dy)
        for q in qs_in
            @test hib(q) === hne(q)
        end
        hext = hermite_interp(x, y, dy; extrap = ExtendExtrap())
        @test isfinite(hext(q_left))
        @test isfinite(hext(q_right))
    end
end

@testitem "UnitStep InBounds locate — persistent hint write-back === NoExtrap (symmetry)" begin
    # Regression: the lean InBounds direct search must still write the persistent hint
    # back to the found interval. It previously dropped the write, so an explicitly
    # provided hint Ref was left STALE under InBounds while NoExtrap/Clamp updated it
    # (per-axis: a mixed `(InBounds, NoExtrap)` query updated only the NoExtrap axis).
    #
    # This pins the PERSISTENT `itp(q; hint)` path (the 6-arg `_search_all_intervals`
    # InBounds overload). The existing oneshot-hint tests use a different search and so
    # never exercised this. The contract is symmetry: InBounds writes the same interval
    # the generic NoExtrap path does — and actually writes it (sentinel 0 must be gone).
    using FastInterpolations: InBounds, NoExtrap, GridIdx

    axx, axy = 1:7, 2:9
    data = [0.1i + 0.3j + 0.01i * j for i in 1:length(axx), j in 1:length(axy)]
    q = (3.4, 6.2)   # in-domain

    @testset "all-axes InBounds writes the hint like NoExtrap — per method" begin
        for ctor in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            itp_ib = ctor((axx, axy), data; extrap = (InBounds(), InBounds()))
            itp_ne = ctor((axx, axy), data)                  # default NoExtrap reference
            hib = (Ref(0), Ref(0))
            hne = (Ref(0), Ref(0))
            itp_ib(q; hint = hib)
            itp_ne(q; hint = hne)
            # chained compare: equal to the NoExtrap interval AND non-sentinel (written).
            @test hib[1][] == hne[1][] != 0
            @test hib[2][] == hne[2][] != 0
        end
    end

    @testset "per-axis mixed: the InBounds axis is not left stale" begin
        for ex in ((InBounds(), NoExtrap()), (NoExtrap(), InBounds()))
            itp = linear_interp((axx, axy), data; extrap = ex)
            ref = linear_interp((axx, axy), data)
            h = (Ref(0), Ref(0))
            hr = (Ref(0), Ref(0))
            itp(q; hint = h)
            ref(q; hint = hr)
            @test h[1][] == hr[1][] != 0   # x axis written (InBounds or NoExtrap)
            @test h[2][] == hr[2][] != 0   # y axis written — the InBounds axis must update too
        end
    end

    @testset "GridIdx InBounds writes the hint like NoExtrap" begin
        itp_ib = linear_interp((axx, axy), data; extrap = (InBounds(), InBounds()))
        itp_ne = linear_interp((axx, axy), data)
        hib = (Ref(0), Ref(0))
        hne = (Ref(0), Ref(0))
        itp_ib((GridIdx(3), 6.2); hint = hib)
        itp_ne((GridIdx(3), 6.2); hint = hne)
        @test hib[1][] == hne[1][] != 0
        @test hib[2][] == hne[2][] != 0
    end
end

@testitem "one-shot InBounds lean search === NoExtrap (all methods, scalar + batch)" begin
    # The one-shot path threads `extraps` into its search (`_search_all_intervals_stencil`
    # for linear/constant, `_search_all_intervals` for cubic/quad), so an InBounds range
    # axis takes the lean `_search_direct_inbounds`. This must be bit-identical to the
    # generic NoExtrap one-shot for in-domain queries — a characterization test (green
    # before and after; the win is perf, verified by benchmark).
    using FastInterpolations: InBounds

    axx, axy = 1:7, 2:9
    data = [0.1i + 0.3j + 0.01i * j for i in 1:length(axx), j in 1:length(axy)]
    qs = ((1.5, 3.5), (3.4, 6.2), (6.99, 8.9), (7.0, 9.0))   # in-domain incl. right boundary
    qxs = Float64[q[1] for q in qs]
    qys = Float64[q[2] for q in qs]
    IBt = (InBounds(), InBounds())

    @testset "scalar one-shot: InBounds === NoExtrap (bit-identical)" begin
        for oneshot in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            for q in qs
                @test oneshot((axx, axy), data, q; extrap = IBt) === oneshot((axx, axy), data, q)
            end
        end
    end

    @testset "batch one-shot: InBounds === NoExtrap (bit-identical)" begin
        for oneshot! in (linear_interp!, cubic_interp!, quadratic_interp!, constant_interp!)
            o_ib = similar(qxs)
            o_ne = similar(qxs)
            oneshot!(o_ib, (axx, axy), data, (qxs, qys); extrap = IBt)
            oneshot!(o_ne, (axx, axy), data, (qxs, qys))
            @test o_ib == o_ne
        end
    end
end

@testitem "one-shot periodic-exclusive seam preserved (InBounds fast path must not touch it)" begin
    # The seam wrap `(n, 1, …)` for `PeriodicBC{:exclusive}` is produced by `search_interval`
    # on `_ExclusivePeriodicAxis` (`<: AbstractVector`, NOT `_CachedRange`) under `WrapExtrap`
    # (never `InBounds`). The lean fast path requires `_CachedRange` + `InBounds`, so it can
    # never match a periodic axis — its `_search_axis_stencil` fallback runs the unchanged
    # code. Pin it: the one-shot at the seam must equal the (unaffected) persistent path.
    using FastInterpolations: PeriodicBC, NoBC

    axx = 1:7
    dataP = [sin(0.4i) + 0.3j for i in 1:7, j in 1:8]
    bcP = (NoBC(), PeriodicBC(endpoint = :exclusive, period = 8.0))
    itp_persist = linear_interp((axx, 1:8), dataP; bc = bcP)

    seam = ((3.5, 8.5), (3.5, 8.99), (1.2, 8.5), (6.8, 8.5))    # all in the seam cell [8, 9)
    @testset "scalar one-shot === persistent at the seam" begin
        for q in seam
            @test linear_interp((axx, 1:8), dataP, q; bc = bcP) ≈ itp_persist(q)
        end
    end
    @testset "batch one-shot === persistent at the seam" begin
        qxs = Float64[q[1] for q in seam]
        qys = Float64[q[2] for q in seam]
        out = similar(qxs)
        linear_interp!(out, (axx, 1:8), dataP, (qxs, qys); bc = bcP)
        for k in eachindex(qxs)
            @test out[k] ≈ itp_persist((qxs[k], qys[k]))
        end
    end
end

@testitem "InBounds on _ExclusivePeriodicAxis routes through the seam, not the lean bypass" begin
    # Regression pin for a latent bug in the InBounds lean work: the generic
    # `search_interval(s, x::AbstractVector, xq, ::InBounds)` overload catches a
    # `_ExclusivePeriodicAxis` and delegates to `_search_interval_real(s, g, xq)` — undefined for
    # the periodic axis (only `g.inner` is searchable; a seam query needs the wrap cell) → a
    # MethodError. A periodic axis is never genuinely InBounds-lean (WrapExtrap semantics), so
    # InBounds must route to the seam-aware 3-arg `search_interval`. Triggered by an INTERIOR query
    # on cubic — its periodic scalar core hoists `_is_inbounds` and passes `InBounds()`, and the ND
    # one-shot OnTheFly collapse builds the `_ExclusivePeriodicAxis` per fiber. The existing seam
    # testitem missed it (linear has no InBounds-periodic branch, and it queried the seam cell only).
    using FastInterpolations: PeriodicBC, WrapExtrap

    x = range(0.0, step = 0.1, length = 20)
    y = range(0.0, step = 0.2, length = 10)
    data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
    bc = PeriodicBC(endpoint = :exclusive, period = 2.0)
    interior = ((0.5, 0.5), (0.05, 0.3), (1.3, 1.1), (0.5, 1.85))   # all strictly in-domain

    # Persistent path searches the periodic axis via the seam method directly (never the 1D-InBounds
    # bypass), so it is unaffected by the bug → use it as the value reference.
    itp_persist = cubic_interp((x, y), data; bc = bc, extrap = WrapExtrap())
    for q in interior
        r = cubic_interp((x, y), data, q; bc = bc, extrap = WrapExtrap())   # pre-fix: MethodError
        @test isfinite(r)
        @test r ≈ itp_persist(q)
    end
end

@testitem "vector (non-uniform) grid InBounds lean === standard (boundary-guard skip)" begin
    # A non-uniform vector grid uses binary search; InBounds drops the `_le(xq,first)` /
    # `_ge(xq,last)` boundary guards (`_search_binary_inbounds`). Bit-identical to NoExtrap
    # for in-domain queries INCLUDING the exact endpoints (where the guards would have
    # early-returned, and the loop returns the same cell). Characterization test; the win is
    # perf. ExtendExtrap on a vector grid stays safe — the generic uses the standard guarded
    # search, and the boundary-guard-free loop clamps OOB naturally anyway.
    using FastInterpolations: InBounds, ExtendExtrap

    xv = [1.0, 1.5, 3.0, 4.2, 7.0, 9.5, 12.0]
    y = [sin(0.3i) + 0.1i for i in 1:length(xv)]
    qs = (1.0, 2.7, 5.5, 9.5, 12.0)             # in-domain incl. both exact endpoints

    @testset "InBounds === NoExtrap (bit-identical), scalar + vector" begin
        for f in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            itp_ib = f(xv, y; extrap = InBounds())
            itp_ne = f(xv, y)
            for q in qs
                @test itp_ib(q) === itp_ne(q)
            end
            qv = collect(range(xv[1], xv[end]; length = 16))
            @test itp_ib(qv) == itp_ne(qv)
        end
    end

    @testset "explicit hint updates under InBounds (vector grid)" begin
        h_ib = Ref(1)
        h_ne = Ref(1)
        linear_interp(xv, y; extrap = InBounds())(5.5; hint = h_ib)
        linear_interp(xv, y)(5.5; hint = h_ne)
        @test h_ib[] == h_ne[] != 1
    end

    @testset "ExtendExtrap OOB on a vector grid stays finite (no lean hazard)" begin
        le = linear_interp(xv, y; extrap = ExtendExtrap())
        @test isfinite(le(0.3))
        @test isfinite(le(13.5))
    end
end

@testitem "InBounds at the exact endpoints x[1]/x[end] === NoExtrap (the dangerous x[end])" begin
    # InBounds treats the exact endpoints as in-domain, so the lean searches MUST return the
    # same bracketing cell as the standard search — in particular `idx == n-1` at `x[end]`
    # (NOT `n`, which would read `x[idx+1]` out of bounds). This pins the contract for both
    # lean searches at both ends:
    #   • range  (`_search_direct_inbounds`): the lean drops only the lower `max(·,1)` and
    #     KEEPS the upper `min(·,n-1)` cap, so `x[end]` clamps to `n-1` even with FP rounding.
    #   • vector (`_search_binary_inbounds`): the binary loop bounds `lo ∈ [1,n-1]`.
    # Covers a non-unit-step range (FP-risky `muladd` near `n`) and a large-offset range
    # (cancellation-risky), plus a vector grid.
    using FastInterpolations: InBounds

    grids = (
        ("range 1:12", 1.0:1.0:12.0),
        ("range(0,10,64)", range(0.0, 10.0; length = 64)),          # non-unit-step h = 10/63
        ("range(1e8,0.1,40)", range(1.0e8; step = 0.1, length = 40)), # large-offset
        ("vector", [1.0, 1.5, 3.0, 4.2, 7.0, 9.5, 12.0]),
    )
    for (lbl, x) in grids
        y = [sin(0.3i) + 0.1i for i in 1:length(x)]
        for f in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            itp_ib = f(x, y; extrap = InBounds())
            itp_ne = f(x, y)
            @test itp_ib(first(x)) === itp_ne(first(x))    # x[1]
            @test itp_ib(last(x)) === itp_ne(last(x))      # x[end] — the dangerous endpoint
        end
        # concrete value pin: linear at x[end] is the right-node value (α = 1 on cell [n-1,n])
        @test linear_interp(x, y; extrap = InBounds())(last(x)) ≈ y[end]
    end
end

@testitem "lean search OOB safety — vector two-sided-safe, range one-sided (contract pin)" begin
    # Direct function-level pin of the search primitives' OOB behavior — the safety contract
    # the NoExtrap/InBounds fast path depends on. Two lean searches, OPPOSITE OOB safety:
    #   • vector `_search_binary_inbounds`: the binary loop keeps `lo ∈ [1, n-1]` for ANY query,
    #     so it returns a valid bracketing cell — and is bit-identical to the guarded
    #     `_search_binary` — even for OOB inputs on BOTH sides. Structurally OOB-safe: promotion
    #     to the lean vector search is safe with or without a preceding domain check.
    #   • range `_search_direct_inbounds`: drops only the lower `max(·,1)` clamp (keeps the upper
    #     `min(·,n-1)` cap). OOB-RIGHT is capped safe; OOB-LEFT returns `idx ≤ 0` (INVALID). This
    #     is the documented hazard, not a bug — it is WHY a lean range search must be reached only
    #     when the query is guaranteed in-domain (NoExtrap's domain check throws first; InBounds /
    #     `@inbounds` is the caller's promise). The guarded `_search_direct` stays two-sided-safe.
    using FastInterpolations: _search_binary, _search_binary_inbounds, _search_direct,
        _search_direct_inbounds

    @testset "vector lean is OOB-safe both sides (=== guarded search)" begin
        xv = [0.0, 0.5, 1.7, 2.2, 3.9, 5.0]
        n = length(xv)
        # OOB-left · exact endpoints · interior grid points · interior · OOB-right
        for xq in (-100.0, -3.0, -1.0e-9, 0.0, 0.25, 2.2, 3.0, 5.0, 5.0 + 1.0e-9, 99.0)
            res = _search_binary_inbounds(xv, xq)
            idx, xL, xR = res
            @test 1 <= idx <= n - 1                     # never idx ≤ 0 or ≥ n
            @test res === _search_binary(xv, xq)        # bit-identical to the guarded search
            @test xL == xv[idx] && xR == xv[idx + 1]
        end
    end

    @testset "range guarded search is OOB-safe both sides" begin
        xr = linear_interp(0.0:1.0:5.0, zeros(6)).x     # _CachedRange
        n = length(xr)
        for xq in (-100.0, -3.0, -1.0e-9, 0.0, 0.25, 5.0, 5.0 + 1.0e-9, 99.0)
            idx, _, _ = _search_direct(xr, xq)
            @test 1 <= idx <= n - 1
        end
    end

    @testset "range lean is one-sided: OOB-right capped, OOB-left INVALID (contract)" begin
        xr = linear_interp(0.0:1.0:5.0, zeros(6)).x
        n = length(xr)
        # in-domain (incl. both endpoints) and OOB-right: valid, upper-capped to n-1
        for xq in (0.0, 0.25, 4.9, 5.0, 5.0 + 1.0e-9, 99.0)
            idx, _, _ = _search_direct_inbounds(xr, xq)
            @test 1 <= idx <= n - 1
        end
        # OOB-LEFT: the dropped lower clamp yields idx ≤ 0. Pinned to lock the asymmetry — a lean
        # range search is only reached under an in-domain guarantee; do not "fix" this to clamp.
        for xq in (-1.0e-9, -3.0, -100.0)
            idx, _, _ = _search_direct_inbounds(xr, xq)
            @test idx <= 0
        end
    end
end

@testitem "lean range search on a _WidenedDomain range keeps the lower clamp (idx ≥ 1)" begin
    # Regression pin: a `_WidenedDomain` `_CachedRange` accepts the widened bracket
    # [domain_lo, domain_hi] ⊋ the grid [lo, hi]. A query in [domain_lo, lo) is IN_DOMAIN (so the
    # domain check / `_oob_state` passes and it reaches the lean search) but sits BELOW the first
    # grid point → arithmetic `idx < 1`. The lean's dropped lower `max(·,1)` would then return
    # `idx ≤ 0` → BoundsError in the caller. The lean MUST keep the two-sided clamp for this tag
    # (fall back to `_search_direct`); only `_UnitStep`/`_Generic` (grid == domain) can drop it.
    using FastInterpolations: _CachedRange, _WidenedDomain, _search_direct_inbounds, _search_direct
    n = 5
    lo = nextfloat(1.0)
    hi = 3.0
    h = (hi - lo) / (n - 1)
    cr = _CachedRange{Float64, Float64, _WidenedDomain}(lo, hi, h, inv(h), n, prevfloat(lo), hi)  # domain_lo = 1.0 < lo
    for xq in (1.0, prevfloat(lo), lo, 1.5, 2.0, hi)   # widened-below · grid-lo · interior · grid-hi
        res = _search_direct_inbounds(cr, xq)
        @test 1 <= res[1] <= n - 1               # never idx ≤ 0 (the widened-region hazard)
        @test res === _search_direct(cr, xq)     # bit-identical to the guarded two-sided clamp
    end
end

@testitem "scalar NoExtrap promotion === InBounds (still throws OOB, Extend still extrapolates)" begin
    # After the domain check passes, a scalar 1D NoExtrap eval promotes to InBounds FOR THE SEARCH
    # (lean guard-free search, coupled to the check: `_check_domain` returns InBounds). This must be
    # bit-identical to an explicit InBounds interpolant on the in-domain contract, while NoExtrap's
    # throw-on-OOB and ExtendExtrap's OOB extrapolation are untouched (Extend passes through the
    # wrapper unchanged and keeps the guarded two-sided-clamp search). Covers all 5 methods on both
    # a uniform range (one-sided range lean) and a non-uniform vector (guard-free binary lean).
    using FastInterpolations: InBounds, ExtendExtrap, NoExtrap

    for (glbl, x) in (("range", 0.0:1.0:10.0), ("vector", [0.0, 0.7, 1.9, 3.3, 5.0, 6.1, 8.4, 10.0]))
        y = [sin(0.4i) + 0.1i for i in 1:length(x)]
        dy = [0.4cos(0.4i) + 0.1 for i in 1:length(x)]
        builders = (
            ("linear", e -> linear_interp(x, y; extrap = e)),
            ("cubic", e -> cubic_interp(x, y; extrap = e)),
            ("quadratic", e -> quadratic_interp(x, y; extrap = e)),
            ("hermite", e -> hermite_interp(x, y, dy; extrap = e)),
            ("constant", e -> constant_interp(x, y; extrap = e)),
        )
        for (mname, build) in builders
            itp_ne = build(NoExtrap())
            itp_ib = build(InBounds())
            itp_ex = build(ExtendExtrap())
            @testset "$mname/$glbl" begin
                for q in (first(x), 2.3, 5.0, 7.777, last(x))      # in-domain incl. exact endpoints
                    @test itp_ne(q) === itp_ib(q)                  # promotion is bit-identical
                end
                @test_throws Exception itp_ne(first(x) - 1.0)      # NoExtrap check survives promotion
                @test_throws Exception itp_ne(last(x) + 1.0)
                @test isfinite(itp_ex(first(x) - 1.0))             # Extend NOT promoted → extrapolates
                @test isfinite(itp_ex(last(x) + 1.0))
            end
        end
    end
end

@testitem "hetero ND PreCompute InBounds lean === NoExtrap (+ ExtendExtrap-OOB safe)" begin
    # Hetero (mixed method per axis) PreCompute paths thread `extraps` into the ND search, so
    # an InBounds range axis takes the lean direct search — bit-identical, per-axis (no 1D
    # shared-core, so an ExtendExtrap axis clamps). Cubic×Linear is a PreCompute hetero.
    using FastInterpolations: InBounds, ExtendExtrap, CubicInterp, LinearInterp

    x = 1.0:1.0:8.0
    y = 2.0:1.0:9.0
    data = [sin(0.3i) + cos(0.2j) + 0.01i * j for i in 1:8, j in 1:8]
    method = (CubicInterp(), LinearInterp())
    qs = ((1.5, 2.5), (4.4, 6.2), (8.0, 9.0))       # in-domain incl. right boundary
    IB = (InBounds(), InBounds())

    @testset "persistent: InBounds === NoExtrap" begin
        itp_ib = interp((x, y), data; method = method, extrap = IB)
        itp_ne = interp((x, y), data; method = method)
        for q in qs
            @test itp_ib(q) === itp_ne(q)
        end
    end

    @testset "one-shot scalar + batch: InBounds === NoExtrap" begin
        for q in qs
            @test interp((x, y), data, q; method = method, extrap = IB) ===
                interp((x, y), data, q; method = method)
        end
        qxs = Float64[q[1] for q in qs]
        qys = Float64[q[2] for q in qs]
        o_ib = similar(qxs)
        o_ne = similar(qxs)
        interp!(o_ib, (x, y), data, (qxs, qys); method = method, extrap = IB)
        interp!(o_ne, (x, y), data, (qxs, qys); method = method)
        @test o_ib == o_ne
    end

    @testset "ExtendExtrap OOB on a hetero axis stays finite (per-axis search clamps)" begin
        itp_ext = interp((x, y), data; method = method, extrap = (ExtendExtrap(), ExtendExtrap()))
        @test isfinite(itp_ext((0.3, 1.5)))
        @test isfinite(itp_ext((9.5, 10.0)))
    end
end
