# ========================================
# Unitful grid axis: 1D families (ducktype-grid phases 2-4)
# ========================================
# Phase 2: Linear + Constant full pipeline (eval, integrate, cumulative,
#          bounded, one-shot) on Quantity grids — unit-asserted outputs.
# Phases 3-4 extend this file per family (cubic/quadratic, then local-slope).

@testitem "Unitful 1D: Linear full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"

    # Real canary: the Float64 twin must stay bit-identical to the phase-0 pin.
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    yf = [0.0, 1.0, 0.5, 2.0, 1.0]
    @test integrate(linear_interp(xf, yf)) === 4.0

    itp = linear_interp(xu, yw)

    @testset "eval: value units, dimensionless weights" begin
        v = itp(1.5u"s")
        @test v === 0.75u"W"
        @test itp(0.0u"s") === 0.0u"W"          # exact node hit
        @test itp(4.0u"s") === 1.0u"W"          # right endpoint
    end

    @testset "integrate: Tg·Tv units" begin
        full = integrate(itp)
        @test full === 4.0u"W*s"
        part = integrate(itp, 0.5u"s", 2.5u"s")
        @test part isa typeof(1.0u"W*s")
        @test part ≈ integrate(linear_interp(xf, yf), 0.5, 2.5) * u"W*s"
    end

    @testset "cumulative_integrate: vector of Tg·Tv" begin
        cum = cumulative_integrate(itp)
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] === integrate(itp)
        @test cum[1] === 0.0u"W*s"
    end

    @testset "one-shot forms" begin
        @test integrate(xu, yw; method = LinearInterp()) === 4.0u"W*s"
        cum = cumulative_integrate(xu, yw; method = LinearInterp())
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] === 4.0u"W*s"
    end

    @testset "range grid" begin
        ru = (0.0:1.0:4.0) .* u"s"
        itr = linear_interp(ru, yw)
        @test itr(1.5u"s") === 0.75u"W"
        @test integrate(itr) === 4.0u"W*s"
    end

    @testset "review pins: one-shot 3-arg scalar, deriv eval" begin
        # F1: one-shot 3-arg scalar (was Tq<:Real — MethodError for Quantity)
        @test linear_interp(xu, yw, 1.5u"s") === 0.75u"W"
        # F3: EvalDeriv1 kernel value-space diff (was coeff-space convert)
        d1 = itp(1.5u"s"; deriv = DerivOp(1))
        @test d1 === -0.5u"W/s"
    end

    @testset "extrapolation modes" begin
        @test_throws DomainError linear_interp(xu, yw)(-1.0u"s")             # NoExtrap OOB
        @test linear_interp(xu, yw; extrap = ClampExtrap())(-1.0u"s") === 0.0u"W"
        @test linear_interp(xu, yw; extrap = ClampExtrap())(9.0u"s") === 1.0u"W"
    end
end

@testitem "Unitful 1D: Constant full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"

    itp = constant_interp(xu, yw)

    @testset "eval" begin
        v = itp(1.4u"s")
        @test v isa typeof(1.0u"W")
    end

    @testset "integrate / cumulative" begin
        full = integrate(itp)
        @test full isa typeof(1.0u"W*s")
        # Constant (nearest) oracle via the Float64 twin
        xf = [0.0, 1.0, 2.0, 3.0, 4.0]
        yf = [0.0, 1.0, 0.5, 2.0, 1.0]
        @test full ≈ integrate(constant_interp(xf, yf)) * u"W*s"
        cum = cumulative_integrate(itp)
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] === full
    end

    @testset "one-shot" begin
        v = integrate(xu, yw; method = ConstantInterp())
        @test v isa typeof(1.0u"W*s")
    end

    @testset "bounded integrate (review pin)" begin
        # F2: impl helper kept x0::Real — MethodError for Quantity bounds
        xf = [0.0, 1.0, 2.0, 3.0, 4.0]
        yf = [0.0, 1.0, 0.5, 2.0, 1.0]
        @test integrate(itp, 0.5u"s", 2.5u"s") ≈
            integrate(constant_interp(xf, yf), 0.5, 2.5) * u"W*s"
    end
end

@testitem "Unitful 1D: dispatch-selection pins (@which)" begin
    using Unitful
    using InteractiveUtils: @which

    # Guards the scalar-vs-batch method split under the unbounded query params:
    # a Quantity scalar must take the scalar arm, a Vector{Quantity} the
    # AbstractVector batch arm (capture here would be invisible to Aqua).
    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"
    itp = linear_interp(xu, yw)

    m_scalar = @which itp(1.5u"s")
    m_batch = @which itp([1.5, 2.5] .* u"s")
    @test m_scalar !== m_batch
    batch_qtype = Base.unwrap_unionall(m_batch.sig).parameters[end]
    scalar_qtype = Base.unwrap_unionall(m_scalar.sig).parameters[end]
    @test batch_qtype <: AbstractVector
    @test !(scalar_qtype <: AbstractArray)
    # batch actually works end-to-end
    out = itp([1.5, 2.5] .* u"s")
    @test out == [0.75, 1.25] .* u"W"
    @test eltype(out) === typeof(1.0u"W")
end

# ========================================
# Phase 3 — Cubic + Quadratic (solver families)
# ========================================

@testitem "Unitful 1D: Cubic full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    yf = [0.0, 1.0, 0.5, 2.0, 1.0]

    itp = cubic_interp(xu, yw)
    tw = cubic_interp(xf, yf)   # Float64 twin (oracle)

    @testset "build: z carries Y/X² units" begin
        @test eltype(itp.z) === typeof(1.0u"W/s^2")
    end

    @testset "eval" begin
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test itp(0.0u"s") ≈ tw(0.0) * u"W" atol = 1.0e-12u"W"
        d1 = itp(1.5u"s"; deriv = DerivOp(1))
        @test d1 ≈ tw(1.5; deriv = DerivOp(1)) * u"W/s"
    end

    @testset "integrate / cumulative / bounded / one-shot" begin
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
        @test integrate(itp, 0.5u"s", 2.5u"s") ≈ integrate(tw, 0.5, 2.5) * u"W*s"
        cum = cumulative_integrate(itp)
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] ≈ integrate(itp)
        @test integrate(xu, yw; method = CubicInterp()) ≈ integrate(tw) * u"W*s"
    end

    @testset "range grid" begin
        ru = (0.0:1.0:4.0) .* u"s"
        @test cubic_interp(ru, yw)(1.5u"s") ≈ tw(1.5) * u"W"
    end
end

@testitem "Unitful 1D: Quadratic full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    yf = [0.0, 1.0, 0.5, 2.0, 1.0]

    itp = quadratic_interp(xu, yw)
    tw = quadratic_interp(xf, yf)

    @testset "build: a=Y/X², d=Y/X coefficient types" begin
        @test eltype(itp.d) === typeof(1.0u"W/s")
        @test eltype(itp.a) === typeof(1.0u"W/s^2")
    end

    @testset "eval / integrate" begin
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
        @test integrate(itp, 0.5u"s", 2.5u"s") ≈ integrate(tw, 0.5, 2.5) * u"W*s"
    end
end
