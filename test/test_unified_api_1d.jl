# ============================================================================
# Unified 1D API: interp(x, y; method=...) / interp!(...)
# ============================================================================
# Goal: a bare-AbstractVector 1D call to the unified API must route to the
# dedicated 1D function and be *identical* to it. Today only the ND forms
# (NTuple grids) exist, so 1D users must write the tuple workaround
# `interp((x,), y; method=(CubicInterp(),))`. These tests pin the desired
# direct form:
#
#     interp(x, y; method=CubicInterp())            === cubic_interp(x, y)
#     interp(x, y, q; method=CubicInterp())         === cubic_interp(x, y, q)
#     interp(x, y, qs; method=CubicInterp())        === cubic_interp(x, y, qs)
#     interp!(out, x, y, qs; method=CubicInterp())  === cubic_interp!(out, x, y, qs)
#
# Every assertion compares the new unified call against the dedicated 1D
# function invoked with the *same* options forwarded from the method struct
# (bc / side / tension), so the tests are self-consistent regardless of
# default values.
#
# RED expectation: all new `interp(::AbstractVector, ::AbstractVector; ...)`
# calls fail with MethodError (no such method) — the first positional arg is
# a Vector, while every existing method requires an NTuple of grids there.
# ============================================================================

@testitem "Unified 1D API: persistent interp(x, y; method=...) routes to dedicated 1D types" begin
    x = collect(range(0.0, 2π, length = 9))
    y = sin.(x)
    q = 1.234

    # Cubic — bc forwarded; result must equal cubic_interp(x, y; bc=...) and be a CubicInterpolant
    @test interp(x, y; method = CubicInterp()) isa CubicInterpolant
    @test interp(x, y; method = CubicInterp())(q) ≈ cubic_interp(x, y; bc = CubicFit())(q) rtol = 1.0e-12
    @test interp(x, y; method = CubicInterp(ZeroCurvBC()))(q) ≈ cubic_interp(x, y; bc = ZeroCurvBC())(q) rtol = 1.0e-12

    # Linear — bc forwarded; result must be a LinearInterpolant
    @test interp(x, y; method = LinearInterp()) isa LinearInterpolant
    @test interp(x, y; method = LinearInterp())(q) ≈ linear_interp(x, y; bc = NoBC())(q) rtol = 1.0e-12

    # Quadratic
    @test interp(x, y; method = QuadraticInterp()) isa QuadraticInterpolant
    @test interp(x, y; method = QuadraticInterp())(q) ≈ quadratic_interp(x, y)(q) rtol = 1.0e-12

    # Constant — side forwarded
    @test interp(x, y; method = ConstantInterp()) isa ConstantInterpolant
    @test interp(x, y; method = ConstantInterp())(q) ≈ constant_interp(x, y)(q) rtol = 1.0e-12
    @test interp(x, y; method = ConstantInterp(LeftSide()))(q) ≈ constant_interp(x, y; side = LeftSide())(q) rtol = 1.0e-12

    # PCHIP
    @test interp(x, y; method = PchipInterp()) isa PchipInterpolant1D
    @test interp(x, y; method = PchipInterp())(q) ≈ pchip_interp(x, y)(q) rtol = 1.0e-12

    # Cardinal — tension forwarded
    @test interp(x, y; method = CardinalInterp()) isa CardinalInterpolant1D
    @test interp(x, y; method = CardinalInterp())(q) ≈ cardinal_interp(x, y)(q) rtol = 1.0e-12
    @test interp(x, y; method = CardinalInterp(tension = 0.5))(q) ≈ cardinal_interp(x, y; tension = 0.5)(q) rtol = 1.0e-12

    # Akima
    @test interp(x, y; method = AkimaInterp()) isa AkimaInterpolant1D
    @test interp(x, y; method = AkimaInterp())(q) ≈ akima_interp(x, y)(q) rtol = 1.0e-12
end

@testitem "Unified 1D API: scalar one-shot interp(x, y, q; method=...)" begin
    x = collect(range(0.0, 2π, length = 9))
    y = sin.(x)
    q = 1.234

    @test interp(x, y, q; method = CubicInterp()) ≈ cubic_interp(x, y, q) rtol = 1.0e-12
    @test interp(x, y, q; method = CubicInterp(ZeroCurvBC())) ≈ cubic_interp(x, y, q; bc = ZeroCurvBC()) rtol = 1.0e-12
    @test interp(x, y, q; method = LinearInterp()) ≈ linear_interp(x, y, q) rtol = 1.0e-12
    @test interp(x, y, q; method = QuadraticInterp()) ≈ quadratic_interp(x, y, q) rtol = 1.0e-12
    @test interp(x, y, q; method = ConstantInterp()) ≈ constant_interp(x, y, q) rtol = 1.0e-12
    @test interp(x, y, q; method = ConstantInterp(LeftSide())) ≈ constant_interp(x, y, q; side = LeftSide()) rtol = 1.0e-12
    @test interp(x, y, q; method = PchipInterp()) ≈ pchip_interp(x, y, q) rtol = 1.0e-12
    @test interp(x, y, q; method = CardinalInterp()) ≈ cardinal_interp(x, y, q) rtol = 1.0e-12
    @test interp(x, y, q; method = CardinalInterp(tension = 0.5)) ≈ cardinal_interp(x, y, q; tension = 0.5) rtol = 1.0e-12
    @test interp(x, y, q; method = AkimaInterp()) ≈ akima_interp(x, y, q) rtol = 1.0e-12
end

@testitem "Unified 1D API: scalar one-shot forwards deriv and extrap" begin
    x = collect(range(0.0, 2π, length = 9))
    y = sin.(x)
    q = 1.234

    # deriv kwarg routed to the dedicated one-shot
    @test interp(x, y, q; method = CubicInterp(), deriv = DerivOp(1)) ≈
        cubic_interp(x, y, q; deriv = DerivOp(1)) rtol = 1.0e-12

    # extrap kwarg routed to the dedicated one-shot (out-of-bounds query)
    q_oob = 10.0
    @test interp(x, y, q_oob; method = CubicInterp(), extrap = ClampExtrap()) ≈
        cubic_interp(x, y, q_oob; extrap = ClampExtrap()) rtol = 1.0e-12
end

@testitem "Unified 1D API: batch allocating interp(x, y, qs; method=...)" begin
    x = collect(range(0.0, 2π, length = 9))
    y = sin.(x)
    qs = [0.5, 1.5, 2.5, 3.0, 5.0]

    @test interp(x, y, qs; method = CubicInterp()) ≈ cubic_interp(x, y, qs) rtol = 1.0e-12
    @test interp(x, y, qs; method = LinearInterp()) ≈ linear_interp(x, y, qs) rtol = 1.0e-12
    @test interp(x, y, qs; method = QuadraticInterp()) ≈ quadratic_interp(x, y, qs) rtol = 1.0e-12
    @test interp(x, y, qs; method = ConstantInterp()) ≈ constant_interp(x, y, qs) rtol = 1.0e-12
    @test interp(x, y, qs; method = PchipInterp()) ≈ pchip_interp(x, y, qs) rtol = 1.0e-12
    @test interp(x, y, qs; method = CardinalInterp()) ≈ cardinal_interp(x, y, qs) rtol = 1.0e-12
    @test interp(x, y, qs; method = AkimaInterp()) ≈ akima_interp(x, y, qs) rtol = 1.0e-12
end

@testitem "Unified 1D API: in-place batch interp!(out, x, y, qs; method=...)" begin
    x = collect(range(0.0, 2π, length = 9))
    y = sin.(x)
    qs = [0.5, 1.5, 2.5, 3.0, 5.0]

    out_new = similar(qs)
    out_ref = similar(qs)

    interp!(out_new, x, y, qs; method = CubicInterp())
    cubic_interp!(out_ref, x, y, qs)
    @test out_new ≈ out_ref rtol = 1.0e-12

    interp!(out_new, x, y, qs; method = LinearInterp())
    linear_interp!(out_ref, x, y, qs)
    @test out_new ≈ out_ref rtol = 1.0e-12

    interp!(out_new, x, y, qs; method = AkimaInterp())
    akima_interp!(out_ref, x, y, qs)
    @test out_new ≈ out_ref rtol = 1.0e-12
end

@testitem "Unified 1D API: allocation parity with dedicated 1D functions" begin
    # The routing must add ZERO allocation: interp(x, y, ...; method=M) must
    # allocate exactly what the dedicated M_interp(x, y, ...) call allocates.
    # Setup + warmup + @allocated all live inside one function (a true function
    # barrier) so locals are concretely typed and @allocated is artifact-free.

    function scalar_parity()
        x = collect(range(0.0, 2π, length = 9)); y = sin.(x); q = 1.234
        interp(x, y, q; method = CubicInterp()); cubic_interp(x, y, q)  # warmup
        a_uni = @allocated interp(x, y, q; method = CubicInterp())
        a_dir = @allocated cubic_interp(x, y, q)
        return a_uni, a_dir
    end

    function inplace_parity()
        x = collect(range(0.0, 2π, length = 9)); y = sin.(x)
        qs = [0.5, 1.5, 2.5, 3.0, 5.0]; out = similar(qs)
        interp!(out, x, y, qs; method = CubicInterp()); cubic_interp!(out, x, y, qs)  # warmup
        a_uni = @allocated interp!(out, x, y, qs; method = CubicInterp())
        a_dir = @allocated cubic_interp!(out, x, y, qs)
        return a_uni, a_dir
    end

    function batch_parity()
        x = collect(range(0.0, 2π, length = 9)); y = sin.(x)
        qs = [0.5, 1.5, 2.5, 3.0, 5.0]
        interp(x, y, qs; method = CubicInterp()); cubic_interp(x, y, qs)  # warmup
        a_uni = @allocated interp(x, y, qs; method = CubicInterp())
        a_dir = @allocated cubic_interp(x, y, qs)
        return a_uni, a_dir
    end

    function persistent_parity()
        x = collect(range(0.0, 2π, length = 9)); y = sin.(x)
        interp(x, y; method = CubicInterp()); cubic_interp(x, y)  # warmup
        a_uni = @allocated interp(x, y; method = CubicInterp())
        a_dir = @allocated cubic_interp(x, y)
        return a_uni, a_dir
    end

    let (a_uni, a_dir) = scalar_parity()
        @test a_uni == a_dir
    end
    let (a_uni, a_dir) = inplace_parity()
        @test a_uni == a_dir   # in-place: both zero after warmup
    end
    let (a_uni, a_dir) = batch_parity()
        @test a_uni == a_dir
    end
    let (a_uni, a_dir) = persistent_parity()
        @test a_uni == a_dir
    end
end

@testitem "Unified 1D API: routing is type-stable (compile-time method resolution)" begin
    using Test: @inferred
    x = collect(range(0.0, 2π, length = 9)); y = sin.(x); q = 1.234
    qs = [0.5, 1.5, 2.5]
    out = similar(qs)

    # `@inferred` throws unless the call is fully type-stable: if the routing were
    # a runtime branch it would produce a `Union` return and fail here. Pairing
    # `@inferred` with `isa typeof(<dedicated call>)` additionally pins that the
    # inferred concrete type is *exactly* the dedicated function's return type —
    # i.e. the wrapper resolves the method at compile time and adds nothing.
    # All 7 families are covered so every `opts` NamedTuple shape is exercised;
    # Constant (side+bc) and Cardinal (bc+tension) are the two-field-`opts` cases
    # most at risk of inference loss through the keyword splat.

    # --- persistent build: returns the dedicated 1D interpolant type ---
    @test (@inferred interp(x, y; method = CubicInterp())) isa typeof(cubic_interp(x, y))
    @test (@inferred interp(x, y; method = LinearInterp())) isa typeof(linear_interp(x, y))
    @test (@inferred interp(x, y; method = QuadraticInterp())) isa typeof(quadratic_interp(x, y))
    @test (@inferred interp(x, y; method = ConstantInterp(LeftSide()))) isa typeof(constant_interp(x, y; side = LeftSide()))
    @test (@inferred interp(x, y; method = PchipInterp())) isa typeof(pchip_interp(x, y))
    @test (@inferred interp(x, y; method = CardinalInterp(tension = 0.5))) isa typeof(cardinal_interp(x, y; tension = 0.5))
    @test (@inferred interp(x, y; method = AkimaInterp())) isa typeof(akima_interp(x, y))

    # --- scalar one-shot: returns the dedicated scalar value type ---
    @test (@inferred interp(x, y, q; method = CubicInterp())) isa typeof(cubic_interp(x, y, q))
    @test (@inferred interp(x, y, q; method = LinearInterp())) isa typeof(linear_interp(x, y, q))
    @test (@inferred interp(x, y, q; method = QuadraticInterp())) isa typeof(quadratic_interp(x, y, q))
    @test (@inferred interp(x, y, q; method = ConstantInterp(LeftSide()))) isa typeof(constant_interp(x, y, q; side = LeftSide()))
    @test (@inferred interp(x, y, q; method = PchipInterp())) isa typeof(pchip_interp(x, y, q))
    @test (@inferred interp(x, y, q; method = CardinalInterp(tension = 0.5))) isa typeof(cardinal_interp(x, y, q; tension = 0.5))
    @test (@inferred interp(x, y, q; method = AkimaInterp())) isa typeof(akima_interp(x, y, q))

    # --- batch allocating + in-place: returns the dedicated Vector / output type ---
    @test (@inferred interp(x, y, qs; method = CubicInterp())) isa typeof(cubic_interp(x, y, qs))
    @test (@inferred interp!(out, x, y, qs; method = CubicInterp())) isa typeof(out)
    @test (@inferred interp!(out, x, y, qs; method = CardinalInterp(tension = 0.5))) isa typeof(out)
end

@testitem "Unified 1D API: unsupported method gives a clear ArgumentError" begin
    # NoInterp and CubicHermiteInterp are exported `AbstractInterpMethod`s with no
    # 1D bare-vector routing (NoInterp is an ND/GridIdx axis marker; CubicHermiteInterp
    # carries no slopes). They must fail with a clear ArgumentError naming the supported
    # methods — not a MethodError on the internal `_interp1d_route`.
    x = collect(range(0.0, 2π, length = 9))
    y = sin.(x)
    q = 1.234
    qs = [0.5, 1.5, 2.5]
    out = similar(qs)

    @test_throws ArgumentError interp(x, y; method = NoInterp())
    @test_throws ArgumentError interp(x, y, q; method = NoInterp())
    @test_throws ArgumentError interp(x, y, qs; method = NoInterp())
    @test_throws ArgumentError interp!(out, x, y, qs; method = NoInterp())
    @test_throws ArgumentError interp(x, y; method = CubicHermiteInterp())
    @test_throws ArgumentError interp(x, y, q; method = CubicHermiteInterp())
end

@testitem "Unified 1D API: bare-vector call equals legacy 1-tuple workaround" begin
    # The direct 1D form must agree with the historical ND-via-1-tuple call.
    x = collect(range(0.0, 2π, length = 9))
    y = sin.(x)
    q = 1.234

    @test interp(x, y; method = CubicInterp())(q) ≈
        interp((x,), reshape(y, :); method = (CubicInterp(),))((q,)) rtol = 1.0e-12
    @test interp(x, y, q; method = LinearInterp()) ≈
        interp((x,), reshape(y, :), (q,); method = (LinearInterp(),)) rtol = 1.0e-12
end
