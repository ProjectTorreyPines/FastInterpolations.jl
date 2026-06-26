# Behavior-pinning tests for the interval-search ordering helpers
# `_le` / `_lt` / `_ge` / `_gt` (src/core/search.jl).
#
# Contract being pinned:
#   1. Each helper is BIT-IDENTICAL to its Base operator (`<=`/`<`/`>=`/`>`) for
#      any operands representable in their common promoted type — i.e. every grid
#      valid for interpolation. (The helpers only `promote`-then-compare to dodge
#      Base's *exact* mixed `<=(Int,Float)`; they never change the result.)
#   2. The `Dual` total order (value tie broken by the partial) is PRESERVED —
#      a primal strip would change it, so this guards that regression.
#   3. Helpers are type-stable `Bool` and allocation-free.
#   4. End-to-end: interval search + one-shot `linear_interp` are identical across
#      `Int` / `Rational` / `Float` grids, and AD through an `Int` grid is correct.

@testitem "Promote-compare ordering helpers" begin
    using FastInterpolations
    using FastInterpolations: _le, _lt, _ge, _gt, _search_binary,
        _oob_state, _is_inbounds, _is_all_inbounds, OOB_LEFT, OOB_RIGHT, IN_DOMAIN
    using ForwardDiff: Dual, derivative

    # ── 1. Equivalence to Base operators ─────────────────────────────────────
    @testset "≡ Base, same + mixed types (within mantissa)" begin
        # integer-valued points: represented exactly by every test type
        ipts = (-7, -3, -1, 0, 1, 3, 7, 12)
        typed = (
            Int64.(ipts), Int32.(ipts), Float64.(ipts),
            Float32.(ipts), Rational{Int}.(ipts),
        )
        for A in typed, B in typed, a in A, b in B
            @test _le(a, b) === (a <= b)
            @test _lt(a, b) === (a < b)
            @test _ge(a, b) === (a >= b)
            @test _gt(a, b) === (a > b)
        end

        # fractional Float query vs Int/Rational grid — the hot mixed case the
        # helpers exist for — tested in BOTH operand orders.
        grids = (Int64[0, 1, 2, 5, 10], Rational{Int}[0, 1, 2, 5, 10])
        for G in grids, g in G, q in -1.0:0.25:11.0
            @test _le(g, q) === (g <= q)
            @test _le(q, g) === (q <= g)
            @test _lt(g, q) === (g < q)
            @test _lt(q, g) === (q < g)
            @test _ge(g, q) === (g >= q)
            @test _ge(q, g) === (q >= g)
            @test _gt(g, q) === (g > q)
            @test _gt(q, g) === (q > g)
        end
    end

    @testset "same-type stays exact beyond 2^53" begin
        # promote is identity for same-type operands → no float rounding, the
        # full integer order is kept even past the Float64 mantissa.
        big = 2^53
        @test _le(big + 1, big + 2) === true
        @test _le(big + 2, big + 1) === false
        @test _lt(big, big) === false
        @test _ge(big + 1, big) === true
        @test _gt(typemax(Int64), typemax(Int64) - 1) === true
        @test _le(typemin(Int64), typemax(Int64)) === true
    end

    # ── 2. Dual total order preserved (NOT a primal strip) ───────────────────
    @testset "Dual value-tie ordering preserved" begin
        d = Dual(5.0, 1.0)   # value 5.0, +partial ⇒ orders ABOVE 5.0 at the tie
        # Pin against Base: a primal strip would make `_le(d, 5.0)` true while
        # `d <= 5.0` is false — these asserts would then fail.
        @test _le(d, 5.0) === (d <= 5.0)
        @test _ge(d, 5.0) === (d >= 5.0)
        @test _lt(5.0, d) === (5.0 < d)
        @test _gt(d, 5.0) === (d > 5.0)
        @test (d <= 5.0) === false          # document the tie-break direction
        @test (d >= 5.0) === true
        # non-tie cases follow value order
        @test _le(d, 6.0) === true
        @test _gt(d, 4.0) === true
        # Dual×Dual, Dual×Int (mixed) all match Base
        e = Dual(5.0, -2.0)
        @test _le(d, e) === (d <= e)
        @test _lt(e, d) === (e < d)
        @test _le(d, 5) === (d <= 5)
        @test _ge(5, d) === (5 >= d)
        # helpers return a plain Bool — no Dual leaks out
        @test _le(d, 6.0) isa Bool
        @test _gt(d, e) isa Bool
    end

    # ── NaN semantics match Base (promote preserves them) ────────────────────
    @testset "NaN matches Base" begin
        for x in (1.0, -1.0, 0.0)
            @test _le(NaN, x) === (NaN <= x)
            @test _lt(x, NaN) === (x < NaN)
            @test _ge(NaN, x) === (NaN >= x)
            @test _gt(NaN, x) === (NaN > x)
        end
        @test _le(NaN, NaN) === false
        @test _le(NaN, 5) === (NaN <= 5)     # mixed NaN/Int
        @test _gt(5, NaN) === (5 > NaN)
    end

    # ── 3. Type stability + zero allocation ──────────────────────────────────
    @testset "type-stable Bool, zero allocation" begin
        @test @inferred(_le(1, 2.0)) isa Bool
        @test @inferred(_lt(1.0, 2)) isa Bool
        @test @inferred(_ge(Int32(1), 2.0)) isa Bool
        @test @inferred(_gt(1 // 2, 0.5)) isa Bool
        @test @inferred(_le(Dual(1.0, 1.0), 2)) isa Bool
        bench() = _le(3, 5.5) & _lt(3, 5.5) & _ge(3, 5.5) & _gt(3, 5.5)
        bench()  # warm up / compile
        @test @allocated(bench()) == 0
    end

    # ── 4. End-to-end equivalence across grid eltypes ────────────────────────
    @testset "interval search identical across grid eltypes" begin
        ints = collect(0:20)                # Vector{Int}
        flts = Float64.(ints)               # reference (Float grid)
        rats = Rational{Int}.(ints)
        qs = vcat(collect(range(-2.0, 22.0; length = 251)), 0.0, 10.0, 20.0)
        for q in qs
            ref = _search_binary(flts, q)[1]     # cell index on the float grid
            @test _search_binary(ints, q)[1] == ref
            @test _search_binary(rats, q)[1] == ref
        end
    end

    @testset "one-shot linear_interp identical: Int vs Float grid" begin
        yv = Float64[i^2 for i in 0:20]
        ints = collect(0:20)
        flts = Float64.(ints)
        # scalar, in-domain + extrapolating policies
        for q in range(-3.0, 23.0; length = 261)
            for ex in (ExtendExtrap(), ClampExtrap())
                @test linear_interp(ints, yv, q; extrap = ex) ==
                    linear_interp(flts, yv, q; extrap = ex)
            end
        end
        # NoExtrap (in-domain only) + batch
        for q in range(0.0, 20.0; length = 151)
            @test linear_interp(ints, yv, q) == linear_interp(flts, yv, q)
        end
        qb = collect(range(0.5, 19.5; length = 53))
        @test linear_interp(ints, yv, qb) == linear_interp(flts, yv, qb)
    end

    @testset "AD derivative through an Int grid" begin
        yv = Float64[i^2 for i in 0:10]    # slope on cell [k,k+1] is 2k+1
        ints = collect(0:10)
        for q0 in (0.5, 3.25, 7.9, 9.5)
            d = derivative(q -> linear_interp(ints, yv, q; extrap = ExtendExtrap()), q0)
            @test d ≈ 2 * floor(Int, q0) + 1
        end
    end

    # ── 5. Domain classification (`_oob_state` / `_is_inbounds`) ──────────────
    # Same promote-compare fix, applied to OOB classification. Both strip the
    # primal of query AND bounds first, so they classify by VALUE (partial-sign
    # independent) — intentionally different from the search ordering helpers,
    # which preserve the Dual total order.
    @testset "classification ≡ across grid eltypes (Int/Float32/Rational/Dual)" begin
        flt = Float64.(collect(0:10))            # reference grid, domain [0, 10]
        ducks = (
            collect(0:10),                        # Int
            collect(0.0f0:1.0f0:10.0f0),          # Float32
            Rational{Int}.(0:10),                 # Rational
            [Dual(Float64(i), 1.0) for i in 0:10],  # Dual grid
        )
        qs = vcat(collect(range(-3.0, 13.0; length = 321)), 0.0, 10.0)  # incl endpoints
        for g in ducks, q in qs
            @test _oob_state(g, q) === _oob_state(flt, q)
            @test _is_inbounds(g, q) === _is_inbounds(flt, q)
        end
        # Rational + Dual queries (duck query types) vs the float grid
        gi = collect(0:10)
        for q in (11 // 2, 21 // 2, -1 // 2, Dual(5.5, 1.0), Dual(-1.0, 1.0), Dual(11.0, 1.0))
            @test _oob_state(gi, q) === _oob_state(flt, q)
            @test _is_inbounds(gi, q) === _is_inbounds(flt, q)
        end
    end

    @testset "classification: explicit values + boundaries" begin
        g = collect(0:10)                        # Int grid, domain [0, 10]
        @test _oob_state(g, -0.5) === OOB_LEFT
        @test _oob_state(g, 10.5) === OOB_RIGHT
        @test _oob_state(g, 5.5) === IN_DOMAIN
        @test _oob_state(g, 0.0) === IN_DOMAIN   # exact endpoints are in-domain
        @test _oob_state(g, 10.0) === IN_DOMAIN
        @test _is_inbounds(g, 0.0) === true
        @test _is_inbounds(g, 10.0) === true
        @test _is_inbounds(g, prevfloat(0.0)) === false
        @test _is_inbounds(g, nextfloat(10.0)) === false
    end

    @testset "classification: Dual is value-based (partial-independent)" begin
        g = collect(0:10)
        @test _oob_state(g, Dual(5.0, 1.0)) === IN_DOMAIN
        @test _oob_state(g, Dual(0.0, -9.0)) === IN_DOMAIN   # boundary, any partial
        @test _oob_state(g, Dual(10.0, 9.0)) === IN_DOMAIN
        @test _oob_state(g, Dual(-0.5, 9.0)) === OOB_LEFT
        @test _oob_state(g, Dual(10.5, -9.0)) === OOB_RIGHT
        @test _is_inbounds(g, Dual(10.0, 5.0)) === true      # in by value
        @test _is_inbounds(g, Dual(10.5, -5.0)) === false
        # Dual grid (differentiate wrt knot positions): bounds primal-stripped too
        dg = [Dual(Float64(i), 1.0) for i in 0:10]
        @test _oob_state(dg, 5.5) === IN_DOMAIN
        @test _oob_state(dg, -1.0) === OOB_LEFT
        @test _is_inbounds(dg, 10.0) === true
    end

    @testset "_is_all_inbounds (batch) ≡ across grid eltypes" begin
        flt = Float64.(collect(0:10))
        ducks = (
            collect(0:10), collect(0.0f0:1.0f0:10.0f0), Rational{Int}.(0:10),
            [Dual(Float64(i), 1.0) for i in 0:10],
        )
        batches = (
            [1.5, 5.0, 9.0],                    # all in
            [-0.5, 5.0],                        # one OOB-left
            [5.0, 10.5],                        # one OOB-right
            [0.0, 10.0],                        # exact endpoints (in)
            Float64[],                          # empty → true
            [Dual(2.0, 1.0), Dual(8.0, 1.0)],   # Dual query batch
        )
        for g in ducks, b in batches
            @test _is_all_inbounds(g, b) === _is_all_inbounds(flt, b)
        end
        gi = collect(0:10)
        @test _is_all_inbounds(gi, [0.0, 10.0]) === true
        @test _is_all_inbounds(gi, [0.0, nextfloat(10.0)]) === false
        @test _is_all_inbounds(gi, [prevfloat(0.0), 5.0]) === false
        @test _is_all_inbounds(gi, Float64[]) === true
    end
end
