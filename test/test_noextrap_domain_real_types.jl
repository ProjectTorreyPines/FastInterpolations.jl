# ═══════════════════════════════════════════════════════════════════════════════
# test_noextrap_domain_real_types.jl
#
# Regression pins for the NoExtrap scalar domain check across Real query types.
#
# The branchless idiom `_clamp(xip, lo, hi) != xip` (utils.jl `_check_domain`)
# silently assumes the in-bounds clamp is a no-op under `==`. That holds only
# when the query and the bounds share a type — `min`/`max` PROMOTE, so the clamp
# result's type can differ from the original query, and `==` across that boundary
# is NOT guaranteed reflexive:
#   * `Irrational`  — `Float64(π) == π` is `false` (exact comparison). ALWAYS red.
#   * `Rational`    — `Float64(1//3) == 1//3` is `false` when inexact. Red.
#   * large `Int`   — `Float64(2^53+1) == 2^53+1` is `false`. Red (rare).
# Floats / small Ints / Duals survive because their `==` promotes both operands
# to a common type first.
#
# A correct in-domain query MUST NOT throw; a correct OOB query MUST throw. The
# Irrational / inexact-Rational interior cases below are GREEN under the old
# comparison check `(xip < lo) || (xip > hi)` and RED under the new clamp-compare.
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "NoExtrap domain check — Real query subtypes" begin
    using ForwardDiff: Dual

    # Two grid representations exercise BOTH _check_domain methods:
    #   range(...)   -> _CachedRange  (clamps against domain_lo/domain_hi)
    #   collect(...) -> Vector        (clamps against first/last)
    big_rng = range(0.0, 10.0, 11)       # interior room for π, ℯ, 1//3, ...
    big_vec = collect(big_rng)
    unit_rng = range(0.0, 1.0, 11)       # π, ℯ, 3.5 land OOB here

    f(g) = sin.(collect(g))
    mk(builder, g) = builder(g, f(g); extrap = NoExtrap())

    # Returns true iff `itp(q)` does NOT raise a DomainError (clean Bool ⇒ a
    # false-positive throw shows as a @test Fail, not an Error). A non-DomainError
    # surfaces unchanged so genuine kernel failures are never masked.
    nothrow(itp, q) = try
        itp(q)
        true
    catch e
        e isa DomainError ? false : rethrow()
    end

    # ── RED PINS: in-domain queries that the clamp-compare wrongly rejects ──
    @testset "Irrational interior — must NOT throw (RED before fix)" begin
        for g in (big_rng, big_vec)
            itp = mk(cubic_interp, g)
            @test nothrow(itp, π)      # 3.14159… ∈ [0,10]
            @test nothrow(itp, ℯ)      # 2.71828… ∈ [0,10]
        end
    end

    @testset "Rational interior — inexact must NOT throw (RED before fix)" begin
        for g in (big_rng, big_vec)
            itp = mk(cubic_interp, g)
            @test nothrow(itp, 1 // 3)   # inexact in Float64 — RED
            @test nothrow(itp, 7 // 3)   # inexact — RED
            @test nothrow(itp, 7 // 2)   # 3.5 exact — already green
        end
    end

    @testset "shared across methods (check lives in one place)" begin
        for builder in (linear_interp, constant_interp, cubic_interp)
            @test nothrow(mk(builder, big_rng), π)
            @test nothrow(mk(builder, big_vec), 1 // 3)
        end
    end

    # ── OOB of the same exotic types still throws (no false negatives) ──
    @testset "Irrational / Rational OOB — must throw" begin
        itp = mk(cubic_interp, unit_rng)
        @test_throws DomainError itp(π)        # 3.14 > 1
        @test_throws DomainError itp(ℯ)        # 2.71 > 1
        @test_throws DomainError itp(7 // 2)   # 3.5 > 1
        @test_throws DomainError itp(-1 // 3)  # < 0
    end

    # ── Regression coverage: types unaffected by the bug stay green ──
    @testset "mixed-precision Float queries — stay green" begin
        itp = mk(cubic_interp, big_rng)
        @test nothrow(itp, 3.5f0)             # Float32 on Float64 grid
        @test nothrow(itp, Float16(3.5))      # Float16
        @test nothrow(itp, big"3.5")          # BigFloat
        # Float64 query on a Float32 grid
        g32 = Float32.(big_vec)
        @test nothrow(cubic_interp(g32, f(g32); extrap = NoExtrap()), 3.5)
        # OOB still throws
        @test_throws DomainError mk(cubic_interp, unit_rng)(3.5f0)
    end

    @testset "Int query — stays green" begin
        itp = mk(cubic_interp, big_rng)
        @test nothrow(itp, 3)                 # interior
        @test_throws DomainError mk(cubic_interp, unit_rng)(3)   # OOB
    end

    @testset "boundary-equal queries — inclusive, must NOT throw" begin
        for g in (big_rng, big_vec)
            itp = mk(cubic_interp, g)
            lo, hi = first(g), last(g)
            @test nothrow(itp, lo)       # exact lower endpoint (Float64)
            @test nothrow(itp, hi)       # exact upper endpoint
            @test nothrow(itp, 0 // 1)   # Rational == lo
            @test nothrow(itp, 10 // 1)  # Rational == hi
        end
    end

    @testset "Dual query — classify by primal, partial sign irrelevant" begin
        itp = mk(cubic_interp, big_rng)
        lo, hi = first(big_rng), last(big_rng)
        # interior primal, either partial sign → in domain
        @test nothrow(itp, Dual(3.5, +1.0))
        @test nothrow(itp, Dual(3.5, -1.0))
        # primal exactly at a boundary, either partial sign → in domain
        @test nothrow(itp, Dual(lo, +1.0))
        @test nothrow(itp, Dual(lo, -1.0))
        @test nothrow(itp, Dual(hi, +1.0))
        @test nothrow(itp, Dual(hi, -1.0))
        # primal OOB → throws regardless of partial sign
        @test_throws DomainError itp(Dual(lo - 0.1, +1.0))
        @test_throws DomainError itp(Dual(hi + 0.1, -1.0))
    end
end
