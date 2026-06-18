# ═══════════════════════════════════════════════════════════════════════════════
# test_query_grid_promotion.jl
#
# Duck-type promotion contract for MISMATCHED query/grid element types.
#
# The package promotes; it must never coerce one side into the other's type.
# Oracle: a type-mismatched query coordinate (Int query on a Float grid, or
# Float query on an Int grid) MUST return the same value as the naturally
# promoted (all-Float) query.
#
# RED at introduction (regression from commit 356259340 / surfaced via #153):
#   `_handle_axis_extrap(q, axis, ::_ClampOrFill)` clamps an OOB query with
#   `oftype(q, first(axis))` — coercing the FLOAT boundary into the QUERY type.
#   For a Float grid with a non-integer endpoint + an Int OOB query this is
#   `convert(Int, 0.5)` → InexactError. Every ND method funnels coordinate
#   clamping through that one shared helper, so all ND methods fail identically
#   under ClampExtrap. 1D and every other extrap were already green and are
#   locked here against future regression.
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "Duck-type query/grid promotion across extraps (1D + ND)" begin
    approx(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-12)

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
    # These pass under the robust `_promote_extrap_val(first(axis), q)` idiom
    # but FAIL (or silently misbehave) under weaker "fixes":
    #   - oftype(q, b)  (the current bug)  → InexactError on Int query / fractional endpoint
    #   - returning `q` unclamped          → fails the exact-boundary-value lock below
    # (The `clamp(q, lo, hi)` variant's failure — snapping `_CachedRange` sliver
    #  queries — is already pinned by test_fillextrap_domain_boundary.jl.)

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
# Green under the current oftype code (Dual queries never trigger InexactError),
# but a forward-looking guard against a weaker fix: the robust `_promote_extrap_val`
# idiom preserves the Dual carrier and zeroes the partial in the flat OOB region.
# A `clamp(q, lo, hi)`-style fix would still pass here on finite endpoints but
# diverges on NaN endpoints / sliver queries — those are pinned elsewhere; this
# locks the core "carrier preserved + zero gradient in the clamped region" contract.
@testitem "ND ClampExtrap — AD carrier preserved, zero gradient in flat region" begin
    using ForwardDiff
    approx(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-12)

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
