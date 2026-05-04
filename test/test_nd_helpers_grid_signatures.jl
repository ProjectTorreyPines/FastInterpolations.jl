@testitem "_search_all_intervals — grid-only overload (no spacings arg)" begin
    using FastInterpolations:
        _search_all_intervals, _ensure_hint_nd, _check_mono_nd,
        AutoSearch, _CachedRange, _CachedVector

    # Setup: 2D Range × Vector with cached wrappers (the post-1D-migration shape)
    rng_grid = FastInterpolations._to_float(0.0:1.0:5.0, Float64)
    vec_grid = FastInterpolations._CachedVector([0.0, 0.5, 1.5, 3.0, 5.0])
    grids = (rng_grid, vec_grid)
    policies = (AutoSearch(), AutoSearch())
    hints = _ensure_hint_nd(nothing, Val(2))
    mono = (false, false)
    q_evals = (2.5, 1.0)

    # New 5-arg overload (no spacings)
    indices, Ls, Rs = _search_all_intervals(q_evals, grids, policies, hints, mono)

    @test indices isa NTuple{2, Int}
    @test Ls isa NTuple{2, <:Real}
    @test Rs isa NTuple{2, <:Real}
    # 2.5 falls in cell 3 of 0:1:5 (between 2.0 and 3.0)
    @test indices[1] == 3
    @test Ls[1] ≈ 2.0
    @test Rs[1] ≈ 3.0
    # 1.0 falls in cell 2 of [0.0, 0.5, 1.5, 3.0, 5.0] (between 0.5 and 1.5)
    @test indices[2] == 2
    @test Ls[2] ≈ 0.5
    @test Rs[2] ≈ 1.5
end

@testitem "_compute_all_local_params — grid-only overload" begin
    using FastInterpolations: _compute_all_local_params, _CachedRange, _CachedVector

    rng_grid = FastInterpolations._to_float(0.0:1.0:5.0, Float64)
    vec_grid = FastInterpolations._CachedVector([0.0, 0.5, 1.5, 3.0, 5.0])
    grids = (rng_grid, vec_grid)
    indices = (3, 2)
    Ls = (2.0, 0.5)
    q_evals = (2.5, 1.0)

    hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids, indices, Ls)

    @test hs[1] ≈ 1.0      # rng cell width
    @test hs[2] ≈ 1.0      # vec cell 2: 1.5 - 0.5
    @test inv_hs[1] ≈ 1.0
    @test inv_hs[2] ≈ 1.0
    @test dLs[1] ≈ 0.5     # 2.5 - 2.0
    @test dLs[2] ≈ 0.5     # 1.0 - 0.5
end

@testitem "_locate_cell_2d_preamble — grid-only overload" begin
    using FastInterpolations:
        _locate_cell_2d_preamble, _ensure_hint_nd, NoExtrap, AutoSearch, _CachedVector

    rng_grid = FastInterpolations._to_float(0.0:1.0:5.0, Float64)
    vec_grid = FastInterpolations._CachedVector([0.0, 0.5, 1.5, 3.0, 5.0])
    grids = (rng_grid, vec_grid)
    extraps = (NoExtrap(), NoExtrap())
    policies = (AutoSearch(), AutoSearch())
    hints = _ensure_hint_nd(nothing, Val(2))
    mono = (false, false)
    query = (2.5, 1.0)

    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, grids, extraps, policies, hints, mono
    )

    @test x_eval ≈ 2.5
    @test y_eval ≈ 1.0
    @test ix == 3
    @test iy == 2
    @test xL ≈ 2.0
    @test yL ≈ 0.5
end
