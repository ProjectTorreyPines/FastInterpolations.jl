# Unit tests for the core-layer _AxisAnchor backbone types: the struct + virtual
# properties (promoted out of gridded so 1D method files can reference them) and
# the generic _StatefulPayload extrap wrapper.

@testitem "_AbstractAnchorPayload marker" begin
    using FastInterpolations: _AbstractAnchorPayload

    @test isabstracttype(_AbstractAnchorPayload)
    @test supertype(_AbstractAnchorPayload) === Any
end

@testitem "all axis payloads share the marker root" begin
    using FastInterpolations: _AbstractAnchorPayload, _StatefulPayload,
        _LinearValuePayload, _LinearDeriv1Payload, _LinearZeroPayload,
        _ConstantValuePayload, _ConstantZeroPayload, _QuadraticPayload,
        _CubicValuePayload1D, _CubicDeriv1Payload1D,
        _CubicDeriv2Payload1D, _CubicDeriv3Payload1D, _CubicZeroPayload1D,
        _LocalHermitePayload, _CubicPartialsPayloadND,
        _QuadraticPartialsPayloadND, IN_DOMAIN

    payload_types = (
        _LinearValuePayload{Float64},
        _LinearDeriv1Payload{Float64, Float64},
        _LinearZeroPayload{Float64},
        _ConstantValuePayload{Float64},
        _ConstantZeroPayload{Float64},
        _QuadraticPayload{Float64},
        _CubicValuePayload1D{Float64},
        _CubicDeriv1Payload1D{Float64},
        _CubicDeriv2Payload1D{Float64},
        _CubicDeriv3Payload1D{Float64},
        _CubicZeroPayload1D{Float64},
        _LocalHermitePayload{Float64, Float64, Float64},
        _CubicPartialsPayloadND{Float64, Float64, Float64},
        _QuadraticPartialsPayloadND{Float64, Float64},
    )

    @test all(P -> P <: _AbstractAnchorPayload, payload_types)
    @test _StatefulPayload{_LinearValuePayload{Float64}} <:
    _AbstractAnchorPayload
    @test_throws MethodError _StatefulPayload((alpha = 0.25,), IN_DOMAIN)
end

@testitem "_StatefulPayload wrapper" begin
    using FastInterpolations: _StatefulPayload, _LinearValuePayload,
        IN_DOMAIN, OOB_LEFT, OOB_RIGHT

    inner = _LinearValuePayload{Float64}(0.25)

    @testset "Construction and fields" begin
        p = _StatefulPayload(inner, IN_DOMAIN)
        @test p.inner === inner
        @test p.state === IN_DOMAIN

        p_left = _StatefulPayload(inner, OOB_LEFT)
        @test p_left.state === OOB_LEFT
        p_right = _StatefulPayload(inner, OOB_RIGHT)
        @test p_right.state === OOB_RIGHT
    end

    @testset "isbits (pool/stream eligibility)" begin
        @test isbitstype(_StatefulPayload{_LinearValuePayload{Float64}})
        @test isbits(_StatefulPayload(inner, IN_DOMAIN))
    end
end

@testitem "_AxisAnchor virtual properties from core layer" begin
    using FastInterpolations: _AxisAnchor, _StatefulPayload, _LinearValuePayload,
        _ContiguousIndices, _ExplicitIndices, _interval_type, IN_DOMAIN, OOB_LEFT

    inner = _LinearValuePayload{Float64}(0.25)

    @testset "Bare payload anchor (pre-move behavior preserved)" begin
        a = _AxisAnchor(_ContiguousIndices{2}(5), inner)
        @test a.idxL == 5
        @test a.idxR == 6
        @test a.alpha == 0.25          # payload-field forwarding
        @test a.payload === inner
    end

    @testset "Stateful payload anchor" begin
        a = _AxisAnchor(_ContiguousIndices{2}(5), _StatefulPayload(inner, OOB_LEFT))
        @test a.idxL == 5
        @test a.idxR == 6
        @test a.state === OOB_LEFT     # forwarded to _StatefulPayload field
        @test a.inner === inner        # forwarded to _StatefulPayload field
        @test isbits(a)
    end

    @testset "Explicit (seam) interval unchanged" begin
        a = _AxisAnchor(_ExplicitIndices(10, 1), _StatefulPayload(inner, IN_DOMAIN))
        @test a.idxL == 10
        @test a.idxR == 1
    end

    @testset "Rejects non-payload structural lookalike" begin
        # A named tuple with an `alpha` property is a structural lookalike, but
        # the `P <: _AbstractAnchorPayload` bound makes the contract nominal.
        @test_throws MethodError _AxisAnchor(
            _ContiguousIndices{2}(5),
            (alpha = 0.25,),
        )
    end

    @testset "_interval_type selector unchanged" begin
        @test _interval_type(collect(1.0:10.0)) === _ContiguousIndices{2}
    end
end
