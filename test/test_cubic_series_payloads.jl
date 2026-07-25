# Unit tests for the cubic Series lean anchor layer: op/extrap-aware payload
# selection, weight bit-identity vs the full `_CubicAdjointAnchor` (same
# `_compute_anchor_weights` must be reused verbatim), and the Series-owned
# anchor build loop (wrap handling, OOB state classification, NoExtrap throw).
# Design: docs/design/cubic_series_payload_anchor.md

@testitem "cubic series payload selector matrix" begin
    using FastInterpolations: _cubic_series_anchor_type, _AxisAnchor, _StatefulPayload,
        _ContiguousIndices, _CubicValuePayload1D, _CubicDeriv1Payload1D,
        _CubicDeriv2Payload1D, _CubicDeriv3Payload1D, _CubicZeroPayload1D,
        _payload_op, _deriv_unit_scale,
        EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3, InBounds

    x = collect(range(0.0, 1.0, 11))
    bare_extraps = (ExtendExtrap(), WrapExtrap(), NoExtrap(), InBounds())
    stateful_extraps = (ClampExtrap(), FillExtrap(0.0))
    # On a Real grid the weight tuple stays the historical NTuple{n,Tq}; only a
    # unit-carrying grid makes the y/z weights dimensionally heterogeneous.
    op_payload_pairs = (
        (EvalValue(), _CubicValuePayload1D{Float64, NTuple{4, Float64}}),
        (EvalDeriv1(), _CubicDeriv1Payload1D{Float64, NTuple{4, Float64}}),
        (EvalDeriv2(), _CubicDeriv2Payload1D{Float64, NTuple{2, Float64}}),
        (EvalDeriv3(), _CubicDeriv3Payload1D{Float64, NTuple{2, Float64}}),
        (DerivOp(5), _CubicZeroPayload1D{Float64}),
    )

    for (op, P) in op_payload_pairs
        # Stateful anchors carry the deriv-unit scale type S (Bool for value, the
        # grid's reciprocal-spacing type for derivatives) — mirror the src formula.
        S = typeof(_deriv_unit_scale(oneunit(eltype(x)), op))
        for e in bare_extraps
            A = @inferred _cubic_series_anchor_type(op, e, x, Float64)
            @test A === _AxisAnchor{_ContiguousIndices{2}, P}
            @test isbitstype(A)
        end
        for e in stateful_extraps
            A = @inferred _cubic_series_anchor_type(op, e, x, Float64)
            @test A === _AxisAnchor{_ContiguousIndices{2}, _StatefulPayload{P, S}}
            @test isbitstype(A)
        end
        # the stateful wrapper forwards its OOB op to the inner payload's op
        @test _payload_op(_StatefulPayload{P, S}) === _payload_op(P)
    end

    # Float32 all the way stays Float32
    x32 = collect(Float32, range(0.0f0, 1.0f0, 11))
    @test _cubic_series_anchor_type(EvalValue(), ExtendExtrap(), x32, Float32) ===
        _AxisAnchor{_ContiguousIndices{2}, _CubicValuePayload1D{Float32, NTuple{4, Float32}}}
end

@testitem "adjoint anchor property ergonomics (idx/idxL/idxR + propertynames)" begin
    using FastInterpolations: _anchor_query

    x = collect(range(0.0, 1.0, 11))
    aq = _anchor_query(x, 0.35, Val(:cubic))   # interior (non-seam) cell

    @test propertynames(aq) ==
        (:interval, :idx, :idxL, :idxR, :xq, :state, :w0, :w1, :w2, :w3)
    # virtual accessors: legacy `idx` == `idxL`, and `idxR == idxL + 1` off-seam
    @test aq.idx == aq.idxL
    @test aq.idxR == aq.idxL + 1
end

@testitem "lean anchor sizes match the design table" begin
    using FastInterpolations: _AxisAnchor, _StatefulPayload, _ContiguousIndices,
        _CubicValuePayload1D, _CubicDeriv1Payload1D, _CubicDeriv2Payload1D,
        _CubicDeriv3Payload1D, _CubicZeroPayload1D

    I2 = _ContiguousIndices{2}
    # On a Real grid the weight tuple `W` stays the historical NTuple{n,Tq}, so the
    # design-table sizes must be untouched by the (unit-grid) heterogeneous split.
    V = _CubicValuePayload1D{Float64, NTuple{4, Float64}}
    D1 = _CubicDeriv1Payload1D{Float64, NTuple{4, Float64}}
    D2 = _CubicDeriv2Payload1D{Float64, NTuple{2, Float64}}
    D3 = _CubicDeriv3Payload1D{Float64, NTuple{2, Float64}}
    Z = _CubicZeroPayload1D{Float64}
    # bare: value/deriv1 40 B, deriv2/3 24 B, zero 8 B
    @test sizeof(_AxisAnchor{I2, V}) == 40
    @test sizeof(_AxisAnchor{I2, D1}) == 40
    @test sizeof(_AxisAnchor{I2, D2}) == 24
    @test sizeof(_AxisAnchor{I2, D3}) == 24
    @test sizeof(_AxisAnchor{I2, Z}) == 8
    # stateful: +state byte + padding → 48/32/16 B. The deriv-unit scale type `S`
    # is a phantom param (no field), so it must not move these sizes.
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{V, Bool}}) == 48
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{D1, Float64}}) == 48
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{D2, Float64}}) == 32
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{D3, Float64}}) == 32
    @test sizeof(_AxisAnchor{I2, _StatefulPayload{Z, Float64}}) == 16
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
            _fill_series_anchors!(FastInterpolations.CubicInterp(), anchors, x, xqs, ExtendExtrap(), false, searcher)
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
    _fill_series_anchors!(FastInterpolations.CubicInterp(), anchors, x, xqs, WrapExtrap(), true, searcher)
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
        result = _fill_series_anchors!(FastInterpolations.CubicInterp(), anchors, x, xqs, extrap, false, searcher)
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
    @test_throws DomainError _fill_series_anchors!(FastInterpolations.CubicInterp(), anchors, x, [0.5, 1.5], NoExtrap(), false, searcher)

    # Mixed precision: Float32 grid + Float64 query must ALSO be DomainError
    # (the typed `_throw_extrap_domain_error` would MethodError here — design §7).
    x32 = collect(Float32, range(0.0f0, 1.0f0, 11))
    searcher32 = _resolve_searcher_for_grid(x32, DEFAULT_SEARCHER)
    A32 = _cubic_series_anchor_type(EvalValue(), NoExtrap(), x32, Float64)
    anchors32 = Vector{A32}(undef, 2)
    err = try
        _fill_series_anchors!(FastInterpolations.CubicInterp(), anchors32, x32, [0.5, 1.5], NoExtrap(), false, searcher32)
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
        @inferred _fill_series_anchors!(FastInterpolations.CubicInterp(), anchors, x, xqs, extrap, false, searcher)
    end
end
