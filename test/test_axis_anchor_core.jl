# Unit tests for the core-layer _AxisAnchor backbone types: the struct + virtual
# properties (promoted out of gridded so 1D method files can reference them) and
# the generic _StatefulPayload extrap wrapper.

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

    @testset "_interval_type selector unchanged" begin
        @test _interval_type(collect(1.0:10.0)) === _ContiguousIndices{2}
    end
end
