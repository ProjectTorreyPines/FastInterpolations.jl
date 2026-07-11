# Inference + zero-allocation pin for the lean-anchor eval delegation. The
# kernel/adapter correctness equivalences (vs the pre-migration full-anchor
# formula) now live in test_cubic_series_lean_batch.jl's persistent_oracle and
# the scalar/point equivalence suites; this file keeps the codegen pin.
# Design: docs/design/cubic_series_payload_anchor.md §5/§7

@testitem "adapter in-domain delegation: inferred and allocation-free" setup = [AllocConstants] begin
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
    @test (@allocated run_mat(Y, Z, anchors, extrap)) <= ALLOC_THRESHOLD
    @test (@allocated run_vec(yv, zv, anchors, extrap)) <= ALLOC_THRESHOLD
end
