# Unit tests for the lean-anchor eval layer: bare payload kernels (both data
# layouts) and the two surface extrap adapters. Oracles are the EXISTING eval
# chains fed the full `_CubicAnchoredQuery` — results must be bit-identical
# (`===`), including each surface's own OOB formula (signed zero and all).
# Design: docs/design/cubic_series_payload_anchor.md §5/§7

@testitem "bare payload kernels === full-anchor kernels (both layouts)" begin
    using FastInterpolations: _cubic_payload_kernel, _cubic_series_anchor_type,
        _fill_series_anchors!, _anchor_query, _eval_series_anchored, _cubic_eval_kernel,
        _resolve_searcher_for_grid, DEFAULT_SEARCHER,
        EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3

    x = cumsum(0.1 .+ rand(11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    xqs = [x[1] + 1.0e-3, 0.5 * (x[5] + x[6]), x[end] - 1.0e-3]
    full = [_anchor_query(x, xq, Val(:cubic)) for xq in xqs]

    Y = rand(11, 3)
    Z = rand(11, 3)
    yv = rand(11)
    zv = rand(11)

    for op in (EvalValue(), EvalDeriv1(), EvalDeriv2(), EvalDeriv3(), DerivOp(5))
        A = _cubic_series_anchor_type(op, ExtendExtrap(), x, Float64)
        anchors = Vector{A}(undef, length(xqs))
        _fill_series_anchors!(anchors, x, xqs, ExtendExtrap(), false, searcher)
        for j in eachindex(xqs), k in 1:3
            @test _cubic_payload_kernel(Y, Z, k, anchors[j]) ===
                _eval_series_anchored(Y, Z, k, full[j], op)
        end
        for j in eachindex(xqs)
            @test _cubic_payload_kernel(yv, zv, anchors[j]) ===
                _cubic_eval_kernel(yv, zv, full[j], op)
        end
    end
end

@testitem "persistent adapter === current _eval_series_with_extrap (incl. OOB formula)" begin
    using FastInterpolations: _cubic_series_eval, _cubic_series_anchor_type,
        _fill_series_anchors!, _anchor_query, _eval_series_with_extrap,
        _resolve_searcher_for_grid, DEFAULT_SEARCHER,
        EvalValue, EvalDeriv1, EvalDeriv2

    x = collect(range(0.0, 1.0, 11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    xqs = [-0.5, 0.5, 1.5]                          # OOB-left, in, OOB-right
    full = [_anchor_query(x, xq, Val(:cubic)) for xq in xqs]

    # -0.0 boundary exercises the persistent `val * one(xq)` signed-zero path
    Y = rand(11, 2)
    Y[1, 1] = -0.0
    Z = rand(11, 2)
    n_pts, x_min, x_max = 11, 0.0, 1.0

    for op in (EvalValue(), EvalDeriv1(), EvalDeriv2(), DerivOp(5))
        for extrap in (ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5))
            A = _cubic_series_anchor_type(op, extrap, x, Float64)
            anchors = Vector{A}(undef, 3)
            _fill_series_anchors!(anchors, x, xqs, extrap, false, searcher)
            for j in 1:3, k in 1:2
                got = _cubic_series_eval(Y, Z, k, anchors[j], extrap)
                want = _eval_series_with_extrap(Y, Z, n_pts, x_min, x_max, k, full[j], extrap, op)
                @test got === want
            end
        end
    end
end

@testitem "one-shot adapter === current _cubic_eval_at_anchor (incl. OOB formula)" begin
    using FastInterpolations: _cubic_series_eval, _cubic_series_anchor_type,
        _fill_series_anchors!, _anchor_query, _cubic_eval_at_anchor,
        _resolve_searcher_for_grid, DEFAULT_SEARCHER,
        EvalValue, EvalDeriv1, EvalDeriv2

    x = collect(range(0.0, 1.0, 11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    xqs = [-0.5, 0.5, 1.5]
    full = [_anchor_query(x, xq, Val(:cubic)) for xq in xqs]

    yv = rand(11)
    yv[1] = -0.0                                     # one-shot normalizes to +0.0 — must match
    zv = rand(11)

    for op in (EvalValue(), EvalDeriv1(), EvalDeriv2(), DerivOp(5))
        for extrap in (ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5))
            A = _cubic_series_anchor_type(op, extrap, x, Float64)
            anchors = Vector{A}(undef, 3)
            _fill_series_anchors!(anchors, x, xqs, extrap, false, searcher)
            for j in 1:3
                got = _cubic_series_eval(yv, zv, anchors[j], extrap)
                want = _cubic_eval_at_anchor(yv, zv, full[j], op, extrap)
                @test got === want
            end
        end
    end
end

@testitem "adapter in-domain delegation: inferred and allocation-free" begin
    using FastInterpolations: _cubic_series_eval, _cubic_series_anchor_type,
        _fill_series_anchors!, _resolve_searcher_for_grid, DEFAULT_SEARCHER, EvalValue

    x = collect(range(0.0, 1.0, 11))
    searcher = _resolve_searcher_for_grid(x, DEFAULT_SEARCHER)
    extrap = ClampExtrap()
    A = _cubic_series_anchor_type(EvalValue(), extrap, x, Float64)
    anchors = Vector{A}(undef, 2)
    _fill_series_anchors!(anchors, x, [0.5, -0.5], extrap, false, searcher)

    Y = rand(11, 2)
    Z = rand(11, 2)
    yv = rand(11)
    zv = rand(11)

    # dedicated pin: if the `_AxisAnchor(interval, inner)` rebuild ever boxes,
    # the streamed-anchor win is negated (design §8)
    run_mat(Y, Z, anchors, extrap) = _cubic_series_eval(Y, Z, 1, anchors[1], extrap)
    run_vec(yv, zv, anchors, extrap) = _cubic_series_eval(yv, zv, anchors[1], extrap)
    @inferred run_mat(Y, Z, anchors, extrap)
    @inferred run_vec(yv, zv, anchors, extrap)
    run_mat(Y, Z, anchors, extrap)
    run_vec(yv, zv, anchors, extrap)
    @test (@allocated run_mat(Y, Z, anchors, extrap)) == 0
    @test (@allocated run_vec(yv, zv, anchors, extrap)) == 0
end
