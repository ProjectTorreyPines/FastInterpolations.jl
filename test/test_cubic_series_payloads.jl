# Unit tests for the cubic Series lean anchor layer: op/extrap-aware payload
# selection, weight bit-identity vs the full `_CubicAnchoredQuery` (same
# `_compute_anchor_weights` must be reused verbatim), and the Series-owned
# anchor build loop (wrap handling, OOB state classification, NoExtrap throw).
# Design: docs/design/cubic_series_payload_anchor.md

@testitem "cubic series payload selector matrix" begin
    using FastInterpolations: _cubic_series_anchor_type, _AxisAnchor, _StatefulPayload,
        _ContiguousIndices, _CubicValuePayload1D, _CubicDeriv1Payload1D,
        _CubicDeriv2Payload1D, _CubicDeriv3Payload1D, _CubicZeroPayload1D,
        EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3, InBounds

    x = collect(range(0.0, 1.0, 11))
    bare_extraps = (ExtendExtrap(), WrapExtrap(), NoExtrap(), InBounds())
    stateful_extraps = (ClampExtrap(), FillExtrap(0.0))
    op_payload_pairs = (
        (EvalValue(), _CubicValuePayload1D{Float64}),
        (EvalDeriv1(), _CubicDeriv1Payload1D{Float64}),
        (EvalDeriv2(), _CubicDeriv2Payload1D{Float64}),
        (EvalDeriv3(), _CubicDeriv3Payload1D{Float64}),
        (DerivOp(5), _CubicZeroPayload1D{Float64}),
    )

    for (op, P) in op_payload_pairs
        for e in bare_extraps
            A = @inferred _cubic_series_anchor_type(op, e, x, Float64)
            @test A === _AxisAnchor{_ContiguousIndices{2}, P}
            @test isbitstype(A)
        end
        for e in stateful_extraps
            A = @inferred _cubic_series_anchor_type(op, e, x, Float64)
            @test A === _AxisAnchor{_ContiguousIndices{2}, _StatefulPayload{P}}
            @test isbitstype(A)
        end
    end

    # Float32 all the way stays Float32
    x32 = collect(Float32, range(0.0f0, 1.0f0, 11))
    @test _cubic_series_anchor_type(EvalValue(), ExtendExtrap(), x32, Float32) ===
        _AxisAnchor{_ContiguousIndices{2}, _CubicValuePayload1D{Float32}}
end

@testitem "lean anchor sizes match the design table" begin
    using FastInterpolations: _AxisAnchor, _StatefulPayload, _ContiguousIndices,
        _CubicValuePayload1D, _CubicDeriv1Payload1D, _CubicDeriv2Payload1D,
        _CubicDeriv3Payload1D, _CubicZeroPayload1D

    I2 = _ContiguousIndices{2}
    # bare: value/deriv1 40 B, deriv2/3 24 B, zero 8 B
    @test sizeof(_AxisAnchor{I2, _CubicValuePayload1D{Float64}}) == 40
    @test sizeof(_AxisAnchor{I2, _CubicDeriv1Payload1D{Float64}}) == 40
    @test sizeof(_AxisAnchor{I2, _CubicDeriv2Payload1D{Float64}}) == 24
    @test sizeof(_AxisAnchor{I2, _CubicDeriv3Payload1D{Float64}}) == 24
    @test sizeof(_AxisAnchor{I2, _CubicZeroPayload1D{Float64}}) == 8
    # stateful: +state byte + padding → 48/32/16 B
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{_CubicValuePayload1D{Float64}}}) == 48
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{_CubicDeriv1Payload1D{Float64}}}) == 48
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{_CubicDeriv2Payload1D{Float64}}}) == 32
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{_CubicDeriv3Payload1D{Float64}}}) == 32
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{_CubicZeroPayload1D{Float64}}}) == 16
end

@testitem "build loop: weights bit-identical to the full anchor oracle" begin
    using FastInterpolations: _cubic_series_anchor_type, _fill_series_anchors!,
        _anchor_query, _resolve_searcher_for_grid, DEFAULT_SEARCHER,
        EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3

    for x in (collect(range(0.0, 1.0, 11)), cumsum(0.1 .+ rand(11)))
        searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
        xqs = [x[1] + 1.0e-3, 0.5 * (x[3] + x[4]), x[end] - 1.0e-3]
        full = [_anchor_query(x, xq, Val(:cubic)) for xq in xqs]

        for (op, wfield) in ((EvalValue(), :w0), (EvalDeriv1(), :w1), (EvalDeriv2(), :w2), (EvalDeriv3(), :w3))
            A = _cubic_series_anchor_type(op, ExtendExtrap(), x, Float64)
            anchors = Vector{A}(undef, length(xqs))
            _fill_series_anchors!(anchors, x, xqs, ExtendExtrap(), false, searcher)
            for j in eachindex(xqs)
                @test anchors[j].w === getproperty(full[j], wfield)
                @test anchors[j].idxL == full[j].idxL
                @test anchors[j].idxR == full[j].idxR
            end
        end
    end
end

@testitem "build loop: WrapExtrap wraps like the full anchor path" begin
    using FastInterpolations: _cubic_series_anchor_type, _fill_series_anchors!,
        _anchor_query, _resolve_searcher_for_grid, DEFAULT_SEARCHER, EvalValue

    x = collect(range(0.0, 1.0, 11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    xqs = [-0.35, 1.45, 0.5]                    # wrap into domain
    full = [_anchor_query(x, xq, Val(:cubic), true) for xq in xqs]

    A = _cubic_series_anchor_type(EvalValue(), WrapExtrap(), x, Float64)
    anchors = Vector{A}(undef, length(xqs))
    _fill_series_anchors!(anchors, x, xqs, WrapExtrap(), true, searcher)
    for j in eachindex(xqs)
        @test anchors[j].w === full[j].w0
        @test anchors[j].idxL == full[j].idxL
    end
end

@testitem "build loop: stateful anchors classify OOB state like the full anchor" begin
    using FastInterpolations: _cubic_series_anchor_type, _fill_series_anchors!,
        _anchor_query, _resolve_searcher_for_grid, DEFAULT_SEARCHER,
        EvalValue, IN_DOMAIN, OOB_LEFT, OOB_RIGHT

    x = collect(range(0.0, 1.0, 11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    xqs = [-0.5, 0.5, 1.5]

    for extrap in (ClampExtrap(), FillExtrap(NaN))
        A = _cubic_series_anchor_type(EvalValue(), extrap, x, Float64)
        anchors = Vector{A}(undef, 3)
        result = _fill_series_anchors!(anchors, x, xqs, extrap, false, searcher)
        @test result === anchors
        @test anchors[1].state === OOB_LEFT
        @test anchors[2].state === IN_DOMAIN
        @test anchors[3].state === OOB_RIGHT
        # in-domain inner weights still match the oracle
        full = _anchor_query(x, 0.5, Val(:cubic))
        @test anchors[2].inner.w === full.w0
    end
end

@testitem "build loop: NoExtrap OOB throws DomainError (mixed precision incl.)" begin
    using FastInterpolations: _cubic_series_anchor_type, _fill_series_anchors!,
        _resolve_searcher_for_grid, DEFAULT_SEARCHER, EvalValue

    # Same precision
    x = collect(range(0.0, 1.0, 11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    A = _cubic_series_anchor_type(EvalValue(), NoExtrap(), x, Float64)
    anchors = Vector{A}(undef, 2)
    @test_throws DomainError _fill_series_anchors!(anchors, x, [0.5, 1.5], NoExtrap(), false, searcher)

    # Mixed precision: Float32 grid + Float64 query must ALSO be DomainError
    # (the typed `_throw_extrap_domain_error` would MethodError here — design §7).
    x32 = collect(Float32, range(0.0f0, 1.0f0, 11))
    searcher32 = _resolve_searcher_for_grid(x32, DEFAULT_SEARCHER)
    A32 = _cubic_series_anchor_type(EvalValue(), NoExtrap(), x32, Float64)
    anchors32 = Vector{A32}(undef, 2)
    err = try
        _fill_series_anchors!(anchors32, x32, [0.5, 1.5], NoExtrap(), false, searcher32)
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test err.val == 1.5                        # offending coordinate carried
end

@testitem "build loop is inference-clean" begin
    using FastInterpolations: _cubic_series_anchor_type, _fill_series_anchors!,
        _resolve_searcher_for_grid, DEFAULT_SEARCHER, EvalValue, EvalDeriv2

    x = collect(range(0.0, 1.0, 11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    xqs = [0.25, 0.75]

    for (op, extrap) in ((EvalValue(), ExtendExtrap()), (EvalDeriv2(), ClampExtrap()))
        A = _cubic_series_anchor_type(op, extrap, x, Float64)
        anchors = Vector{A}(undef, 2)
        @inferred _fill_series_anchors!(anchors, x, xqs, extrap, false, searcher)
    end
end
