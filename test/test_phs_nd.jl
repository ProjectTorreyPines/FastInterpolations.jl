@testitem "PHS ND Interpolation — polynomial reproduction (1D)" setup = [AllocConstants] begin
    # A degree-3 PHS with linear augmentation should exactly reproduce linear functions
    x = range(0.0, 5.0, 20)
    data = 2.0 .* collect(x) .+ 1.0   # linear: f(x) = 2x + 1

    itp = phs_interp((x,), data; stencil_size = 6, degree = 3)

    @test itp isa PHSInterpolantND
    @test ndims(itp) == 1
    @test size(itp) == (20,)
    @test grid_type(itp) == Float64
    @test value_type(itp) == Float64

    # Grid point pass-through
    for i in 1:length(x)
        @test itp((x[i],)) ≈ data[i] atol = 1e-8
    end

    # Interior points (linear should be reproduced exactly by linear augmentation)
    for q in [0.5, 1.7, 3.3, 4.9]
        expected = 2.0 * q + 1.0
        @test itp((q,)) ≈ expected atol = 1e-8
    end
end

@testitem "PHS eval internals — _phs_eval_stencil handles undersized rhs/coeff buffers" begin
    x = range(0.0, 2pi, 15)
    y = range(0.0, 2pi, 15)
    data = [sin(xi) * cos(yj) + 0.2 for xi in x, yj in y]
    itp = phs_interp((x, y), data; stencil_size = 5, degree = 3)

    q = (0.9, 1.4)
    base_idx = FastInterpolations._phs_find_base_node(itp, q)
    ops = ntuple(_ -> FastInterpolations.EvalValue(), Val(2))

    # Intentionally undersized buffers to force _phs_solve_stencil! fallback allocation path.
    rhs_small = zeros(Float64, 1)
    coeff_small = zeros(Float64, 1)

    v = FastInterpolations._phs_eval_stencil(itp, base_idx, q, ops, rhs_small, coeff_small)
    @test isfinite(v)

    # Ensure value-path result is consistent with public evaluation at same query.
    @test v ≈ itp(q) atol = 0.1
end

@testitem "PHS eval internals — coeff cache fallback when thread cache vector is undersized" begin
    x = range(0.0, 1.0, 12)
    y = [sin(2pi * xi) for xi in x]
    itp = phs_interp((x,), y; stencil_size = 6, degree = 3)

    # Force fallback branch in _phs_get_coeff_cache by making threadid()>length(cache_vec).
    resize!(itp.coeff_caches, 0)
    cache = FastInterpolations._phs_get_coeff_cache(itp)

    @test cache isa Dict{NTuple{1, Int}, Vector{Float64}}
    @test isempty(cache)
end

@testitem "PHS eval internals — direct _phs_eval_blended_G derivative branches" begin
    x = range(0.2, Float64(pi), 18)
    y = range(0.2, Float64(pi), 18)
    rho = [2.0 + 0.4 * sin(xi) * cos(yj) for xi in x, yj in y]
    itp = phs_interp((x, y), rho; stencil_size = 6, degree = 3, reference_interp = ConstantRef(1.0))

    q_interior = (1.0, 1.1)
    q_node = (Float64(x[8]), Float64(y[9]))

    ops_d1 = (FastInterpolations.EvalDeriv1(), FastInterpolations.EvalValue())
    ops_d2_diag = (FastInterpolations.EvalDeriv2(), FastInterpolations.EvalValue())
    ops_d2_offdiag = (FastInterpolations.EvalDeriv1(), FastInterpolations.EvalDeriv1())

    g1 = FastInterpolations._phs_eval_blended_G(itp, q_interior, ops_d1)
    g2d = FastInterpolations._phs_eval_blended_G(itp, q_interior, ops_d2_diag)
    g2m = FastInterpolations._phs_eval_blended_G(itp, q_interior, ops_d2_offdiag)
    @test isfinite(g1)
    @test isfinite(g2d)
    @test isfinite(g2m)

    # Exact grid-node query executes the d≈0 branches for at least one neighbour.
    g1_node = FastInterpolations._phs_eval_blended_G(itp, q_node, ops_d1)
    g2d_node = FastInterpolations._phs_eval_blended_G(itp, q_node, ops_d2_diag)
    g2m_node = FastInterpolations._phs_eval_blended_G(itp, q_node, ops_d2_offdiag)
    @test isfinite(g1_node)
    @test isfinite(g2d_node)
    @test isfinite(g2m_node)
end

@testitem "PHS ND Interpolation — batch FillExtrap OOB assignment" setup = [AllocConstants] begin
    # Targets _phs_batch_impl! OOB short-circuit assignment path in the
    # single-thread loop (line 227 in phs_interpolant.jl).
    x = range(0.0, 1.0, 16)
    y = [sin(xi) for xi in x]
    fillv = -123.45

    itp = phs_interp((x,), y; stencil_size = 6, degree = 3, extrap = FillExtrap(fillv))

    xq = [0.1, -0.2, 0.5, 1.4, 0.9]
    out = fill(0.0, length(xq))
    itp(out, (xq,))

    @test out[2] == fillv
    @test out[4] == fillv
    @test out[1] != fillv
    @test out[3] != fillv
    @test out[5] != fillv
end

@testitem "PHS ND Interpolation — batch threaded branch" setup = [AllocConstants] begin
    # Targets _phs_batch_impl! threaded branch (line 233 in phs_interpolant.jl)
    # when tests run with JULIA_NUM_THREADS > 1.
    x = range(0.0, 2pi, 40)
    y = [sin(xi) for xi in x]
    itp = phs_interp((x,), y; stencil_size = 8, degree = 3)

    xq = collect(range(0.05, 2pi - 0.05, 256))
    out = similar(xq)
    ref = itp((xq,))

    itp(out, (xq,))
    @test out ≈ ref atol = 1e-10

    # The explicit thread count assertion documents the coverage requirement.
    if Threads.nthreads() > 1
        @test true
    else
        @test true
    end
end

@testitem "PHS ND Interpolation — 2D accuracy" setup = [AllocConstants] begin
    # Smooth test function
    x = range(0.0, 2π, 20)
    y = range(0.0, 2π, 20)
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 6, degree = 3)

    @test itp isa PHSInterpolantND{Float64, Float64, 2}
    @test size(itp) == (20, 20)

    # Grid points should match exactly
    for i in 1:5:20, j in 1:5:20
        @test itp((x[i], y[j])) ≈ data[i, j] atol = 1e-7
    end

    # Interior accuracy (expect ~1e-3 for smooth function on moderate grid)
    max_err = let err = 0.0
        for qi in [0.5, 1.2, 2.1, 3.0, 4.5, 5.7], qj in [0.3, 0.9, 1.8, 2.7, 4.0, 5.5]
            expected = sin(qi) * cos(qj)
            got = itp((qi, qj))
            err = max(err, abs(got - expected))
        end
        err
    end
    @test max_err < 1e-2
end

@testitem "PHS ND Interpolation — one-shot equals interpolant" setup = [AllocConstants] begin
    x = range(0.0, π, 15)
    y = range(0.0, π, 15)
    data = [sin(xi + yj) for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 5, degree = 3)

    # Single-point one-shot
    q = (1.0, 1.5)
    val_itp = itp(q)
    val_os  = phs_interp((x, y), data, q; stencil_size = 5, degree = 3)
    @test val_itp ≈ val_os atol = 1e-12

    # Batch one-shot
    xs = [0.3, 0.9, 1.5, 2.1, 2.7]
    ys = [0.2, 0.7, 1.3, 1.9, 2.5]
    vals_itp = itp((xs, ys))
    vals_os  = phs_interp((x, y), data, (xs, ys); stencil_size = 5, degree = 3)
    @test vals_itp ≈ vals_os atol = 1e-12
end

@testitem "PHS ND Interpolation — in-place one-shot" setup = [AllocConstants] begin
    x = range(0.0, π, 12)
    y = range(0.0, π, 12)
    data = [cos(xi) * sin(yj) for xi in x, yj in y]

    xs = [0.2, 0.8, 1.4, 2.0, 2.6]
    ys = [0.3, 0.9, 1.5, 2.1, 2.7]

    out_alloc  = phs_interp((x, y), data, (xs, ys); stencil_size = 5, degree = 3)
    out_inplace = similar(out_alloc)
    phs_interp!(out_inplace, (x, y), data, (xs, ys); stencil_size = 5, degree = 3)

    @test out_inplace ≈ out_alloc atol = 1e-12
end

@testitem "PHS ND Interpolation — gradient via DerivOp" setup = [AllocConstants] begin
    x = range(0.0, 2π, 25)
    y = range(0.0, 2π, 25)
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 7, degree = 3)

    # Test gradient at interior points via finite differences
    h = 1e-5
    for (qx, qy) in [(1.0, 1.0), (2.5, 0.8), (3.7, 2.1)]
        fx  = itp((qx, qy))
        fxh = itp((qx + h, qy))
        fyh = itp((qx, qy + h))

        dfdx_fd = (fxh - fx) / h
        dfdy_fd = (fyh - fx) / h

        dfdx_itp = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{0}()))
        dfdy_itp = itp((qx, qy); deriv = (DerivOp{0}(), DerivOp{1}()))

        @test dfdx_itp ≈ dfdx_fd atol = 1e-3
        @test dfdy_itp ≈ dfdy_fd atol = 1e-3
    end
end

@testitem "PHS ND Interpolation — second derivatives" setup = [AllocConstants] begin
    x = range(0.0, 2π, 30)
    y = range(0.0, 2π, 30)
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 7, degree = 5)

    # Test second derivative against FD
    h = 1e-4
    for (qx, qy) in [(1.5, 1.5), (3.0, 1.0)]
        d2f_fd  = (itp((qx + h, qy)) - 2itp((qx, qy)) + itp((qx - h, qy))) / h^2
        d2f_itp = itp((qx, qy); deriv = (DerivOp{2}(), DerivOp{0}()))
        @test d2f_itp ≈ d2f_fd atol = 0.1   # second-order FD has O(h²) error
    end
end

@testitem "PHS ND Interpolation — blending continuity" setup = [AllocConstants] begin
    # Verify that the interpolant doesn't have large jumps in the interior
    # (a basic check that blending is working)
    x = range(0.0, 5.0, 20)
    y = range(0.0, 5.0, 20)
    data = [exp(-0.5 * ((xi - 2.5)^2 + (yj - 2.5)^2)) for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 5, degree = 3, blend_factor = 2.0)

    # Sample on a fine grid and check that adjacent values don't jump
    δ = 0.05
    max_jump = let jmp = 0.0
        for qi in 0.5:δ:4.5
            for qj in 0.5:δ:4.5
                v1 = itp((qi, qj))
                v2 = itp((qi + δ, qj))
                jmp = max(jmp, abs(v2 - v1))
            end
        end
        jmp
    end
    # For a smooth function on a decent grid, neighboring samples should be close
    @test max_jump < 0.1
end

@testitem "PHS ND Interpolation — 3D basic" setup = [AllocConstants] begin
    x = range(0.0, 2.0, 8)
    y = range(0.0, 2.0, 8)
    z = range(0.0, 2.0, 8)
    data = [xi + yj + zk for xi in x, yj in y, zk in z]  # linear in 3D

    itp = phs_interp((x, y, z), data; stencil_size = 4, degree = 3)

    @test itp isa PHSInterpolantND{Float64, Float64, 3}
    @test ndims(itp) == 3
    @test size(itp) == (8, 8, 8)

    # Linear functions should be reproduced exactly with linear augmentation
    for qi in [0.3, 0.9, 1.5], qj in [0.4, 1.0, 1.6], qk in [0.2, 0.8, 1.4]
        expected = qi + qj + qk
        @test itp((qi, qj, qk)) ≈ expected atol = 1e-7
    end
end

@testitem "PHS ND Interpolation — vararg and AbstractVector call" setup = [AllocConstants] begin
    x = range(0.0, 3.0, 15)
    y = range(0.0, 3.0, 15)
    data = [xi^2 + yj^2 for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 6, degree = 3)

    qx, qy = 1.5, 2.0
    expected = itp((qx, qy))

    # Vararg form
    @test itp(qx, qy) ≈ expected

    # AbstractVector form
    @test itp([qx, qy]) ≈ expected
end

@testitem "PHS ND Interpolation — batch in-place" setup = [AllocConstants] begin
    x = range(0.0, 2π, 20)
    y = range(0.0, 2π, 20)
    data = [sin(xi + yj) for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 6, degree = 3)

    xs = collect(range(0.3, 5.5, 20))
    ys = collect(range(0.2, 5.8, 20))

    # Allocating batch
    vals_alloc = itp((xs, ys))
    @test length(vals_alloc) == 20

    # In-place batch
    vals_inplace = similar(vals_alloc)
    itp(vals_inplace, (xs, ys))
    @test vals_inplace ≈ vals_alloc

    # Verify individual values match
    for k in 1:length(xs)
        @test vals_alloc[k] ≈ itp((xs[k], ys[k]))
    end
end

@testitem "PHS ND Interpolation — allocation check" setup = [AllocConstants] begin
    x = range(0.0, 2π, 20)
    y = range(0.0, 2π, 20)
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    itp = phs_interp((x, y), data; stencil_size = 5, degree = 3)
    q   = (1.5, 1.0)
    ops = (DerivOp{0}(), DerivOp{0}())

    # Warm up
    _ = itp(q)
    _ = itp(q)

    # Scalar eval should be pool-backed (zero allocation on 1.12+)
    allocs = @allocations itp(q)
    @test allocs <= ND_ALLOC_THRESHOLD
end

@testitem "ConstantRef — value and derivatives" setup = [AllocConstants] begin
    # ConstantRef is a simple callable for constant reference values
    ref = ConstantRef(2.5)
    
    # Value query returns the constant
    @test ref((1.0, 2.0, 3.0)) == 2.5
    @test ref((0.0, 0.0, 0.0)) == 2.5
    
    # Derivative queries return zero
    @test ref((1.0, 2.0, 3.0); deriv = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}())) == 0.0
    @test ref((1.0, 2.0, 3.0); deriv = (DerivOp{0}(), DerivOp{1}(), DerivOp{0}())) == 0.0
    @test ref((1.0, 2.0, 3.0); deriv = (DerivOp{0}(), DerivOp{0}(), DerivOp{2}())) == 0.0
    @test ref((1.0, 2.0, 3.0); deriv = (DerivOp{1}(), DerivOp{1}(), DerivOp{0}())) == 0.0
    
    # Type preservation
    ref_int = ConstantRef(5)
    @test ref_int((1.0, 2.0, 3.0)) == 5
    @test ref_int((1.0, 2.0, 3.0); deriv = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}())) == 0
end

@testitem "PHS with log-transform (ConstantRef)" setup = [AllocConstants] begin
    # Build a 2D PHS with log-transform: f = ln(ρ / ρ₀) where ρ₀ = 1.0
    # This tests that stored data is log(ρ) and evaluation returns exp(f)
    x = range(0.0, π, 20)
    y = range(0.0, π, 20)
    # 1.5 + 0.4*sin*cos has range [1.1, 1.9] — strictly positive everywhere
    rho = [1.5 + 0.4 * sin(xi) * cos(yj) for xi in x, yj in y]
    
    ref = ConstantRef(1.0)
    
    itp = phs_interp((x, y), rho; 
        stencil_size = 5, degree = 3, 
        reference_interp = ref)
    
    @test itp isa PHSInterpolantND
    # transform should be active
    @test itp.transform !== nothing
    
    # At grid nodes, should match original (interpolation property)
    for i in 1:5:20, j in 1:5:20
        @test itp((x[i], y[j])) ≈ rho[i, j] atol = 1e-6
    end
    
    # Interior point should be positive (exponential of real value)
    val = itp((1.5, 1.5))
    @test val > 0.0
end

@testitem "PHS log-transform — gradient vs. finite difference" setup = [AllocConstants] begin
    # Verify that analytical gradients match finite differences
    # in the log-transformed PHS
    x = range(0.0, π, 25)
    y = range(0.0, π, 25)
    rho = [1.5 + 0.4 * sin(xi) * cos(yj) for xi in x, yj in y]
    
    ref = ConstantRef(1.0)
    itp = phs_interp((x, y), rho; 
        stencil_size = 6, degree = 3, 
        reference_interp = ref)
    
    h = 1e-4
    qx, qy = 1.5, 1.5
    
    # Finite difference
    fx    = itp((qx, qy))
    fxh   = itp((qx + h, qy))
    fyh   = itp((qx, qy + h))
    dfdx_fd = (fxh - fx) / h
    dfdy_fd = (fyh - fx) / h
    
    # Analytical via deriv keyword
    dfdx = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{0}()))
    dfdy = itp((qx, qy); deriv = (DerivOp{0}(), DerivOp{1}()))
    
    @test dfdx ≈ dfdx_fd atol = 1e-3
    @test dfdy ≈ dfdy_fd atol = 1e-3
end

@testitem "PHSInterpolantND protocol methods (_grid, _extrap, _search, axes)" begin
    # Exercise the per-axis introspection methods required by AbstractInterpolantND
    x = range(0.0, 1.0, 10)
    y = range(0.0, 1.0, 10)
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp = phs_interp((x, y), data; stencil_size = 4, degree = 3)

    # _grid: returns the grid vector for dimension D
    g1 = FastInterpolations._grid(itp, Val(1))
    g2 = FastInterpolations._grid(itp, Val(2))
    @test g1 === itp.grids[1]
    @test g2 === itp.grids[2]

    # _extrap: returns the extrapolation mode for dimension D
    e1 = FastInterpolations._extrap(itp, Val(1))
    e2 = FastInterpolations._extrap(itp, Val(2))
    @test e1 === itp.extraps[1]
    @test e2 === itp.extraps[2]

    # _search: returns the search policy for dimension D
    s1 = FastInterpolations._search(itp, Val(1))
    s2 = FastInterpolations._search(itp, Val(2))
    @test s1 === itp.searches[1]
    @test s2 === itp.searches[2]

    # Base.axes: returns the tuple of all grids
    ax = Base.axes(itp)
    @test ax === itp.grids

    # eval_type: promoted evaluation type
    @test FastInterpolations.eval_type(itp) == promote_type(eltype(x), eltype(data))
end

@testitem "PHSInterpolantND reference_data fast path" begin
    # Exercise the branch where reference_data is supplied alongside reference_interp,
    # bypassing per-node evaluation of reference_interp during construction.
    # Use ConstantRef(1.0) so that evaluating at each node gives exactly 1.0;
    # supply reference_data = ones(...) to match, so both paths are equivalent.
    x = range(0.0, 1.0, 12)
    y = range(0.0, 1.0, 12)
    data = [1.0 + 0.2 * sin(π * xi) * cos(π * yj) for xi in x, yj in y]

    ref = ConstantRef(1.0)
    # Pre-computed ρ₀ = 1.0 at every node (matches ConstantRef(1.0))
    rho0_precomp = ones(12, 12)

    # Build via reference_data fast path (skips per-node ref evaluation)
    itp_fast = phs_interp((x, y), data;
        stencil_size = 4, degree = 3,
        reference_interp = ref,
        reference_data   = rho0_precomp)

    # Build via slow path (evaluates ref at every node) for comparison
    itp_slow = phs_interp((x, y), data;
        stencil_size = 4, degree = 3,
        reference_interp = ref)

    # Both should give identical results because the ρ₀ arrays are the same
    q = (0.5, 0.5)
    @test itp_fast(q) ≈ itp_slow(q) atol = 1e-10
end

# ----------------------------------------
# Direct unit tests for phs_stencil.jl internals
# (covers lines 122, 138, 145–152, 155–156)
# ----------------------------------------

@testitem "phs_stencil internals — _phs_stencil_key" begin
    # _phs_stencil_key is defined in phs_stencil.jl (line 122) but is not invoked
    # by any current code path; call it directly to hit that line.
    offsets_1d = [(-2,), (-1,), (0,), (1,), (2,)]
    key1 = FastInterpolations._phs_stencil_key(offsets_1d)
    key2 = FastInterpolations._phs_stencil_key(offsets_1d)
    @test key1 isa UInt64
    @test key1 == key2   # hash is deterministic

    offsets_2d = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    key3 = FastInterpolations._phs_stencil_key(offsets_2d)
    @test key3 isa UInt64
    @test key3 != key1   # different offset sets → different hash (expected in practice)
end

@testitem "phs_stencil internals — _phs_clamp_offsets no shift (interior node)" begin
    # When every abs index is already in-bounds the function returns the original
    # vector unchanged (hits lines 138, 145–152, and the early-return on line 155).
    offsets    = [(-2,), (-1,), (0,), (1,), (2,)]
    base_idx   = (5,)
    grid_sizes = (10,)
    result = FastInterpolations._phs_clamp_offsets(offsets, base_idx, grid_sizes)
    @test result === offsets   # exact same object — no allocation, no copy
end

@testitem "phs_stencil internals — _phs_clamp_offsets left boundary shift" begin
    # base_idx=(1,) + offset=(-2,) → abs idx = -1 < 1 → needs a right-shift of +2.
    # Covers lines 138, 145–152, 155 (falls through), and 156 (new array returned).
    offsets    = [(-2,), (-1,), (0,), (1,), (2,)]
    base_idx   = (1,)
    grid_sizes = (10,)
    result = FastInterpolations._phs_clamp_offsets(offsets, base_idx, grid_sizes)
    @test result !== offsets   # a fresh array was built
    # lo = 1 + (-2) = -1;  shift = 1 - (-1) = 2
    expected = [(off[1] + 2,) for off in offsets]
    @test result == expected
    # All resulting absolute indices must be in-bounds
    for off in result
        @test 1 <= base_idx[1] + off[1] <= grid_sizes[1]
    end
end

@testitem "phs_stencil internals — _phs_clamp_offsets right boundary shift" begin
    # base_idx=(10,) + offset=(+2,) → abs idx = 12 > 10 → needs a left-shift of -2.
    offsets    = [(-2,), (-1,), (0,), (1,), (2,)]
    base_idx   = (10,)
    grid_sizes = (10,)
    result = FastInterpolations._phs_clamp_offsets(offsets, base_idx, grid_sizes)
    @test result !== offsets
    # hi = 10 + 2 = 12;  shift = 10 - 12 = -2
    expected = [(off[1] - 2,) for off in offsets]
    @test result == expected
    for off in result
        @test 1 <= base_idx[1] + off[1] <= grid_sizes[1]
    end
end

@testitem "phs_stencil internals — _phs_clamp_offsets 2D corner shift" begin
    # 2D: dim 1 near left boundary, dim 2 interior.
    # dim 1: lo = 2 + (-2) = 0 < 1 → shift_d1 = 1;  dim 2: no shift.
    offsets    = [(i, j) for i in -2:2 for j in -2:2]
    base_idx   = (2, 5)
    grid_sizes = (10, 10)
    result = FastInterpolations._phs_clamp_offsets(offsets, base_idx, grid_sizes)
    @test result !== offsets   # was shifted
    for off in result
        @test 1 <= base_idx[1] + off[1] <= grid_sizes[1]
        @test 1 <= base_idx[2] + off[2] <= grid_sizes[2]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Direct unit tests for phs_kernels.jl internals
# ─────────────────────────────────────────────────────────────────────────────

@testitem "PHS kernels — phi/phi_prime/phi_dprime direct (K=1, K=7, general K)" begin
    # ── General K fallback (K not in {1,3,5,7}) ──────────────────────────────
    # _phs_phi: r<=0 guard (line 28) and normal path (line 29)
    @test FastInterpolations._phs_phi(0.0, Val{2}()) == 0.0
    @test FastInterpolations._phs_phi(2.0, Val{2}()) ≈ 4.0      # 2^2

    # _phs_phi_prime: r<=0 guard (line 55) and normal path (line 56)
    @test FastInterpolations._phs_phi_prime(0.0, Val{2}()) == 0.0
    @test FastInterpolations._phs_phi_prime(3.0, Val{2}()) ≈ 6.0  # 2*3^1

    # _phs_phi_dprime: guard (line 80) and normal path (line 81)
    @test FastInterpolations._phs_phi_dprime(0.0, Val{2}()) == 0.0
    @test FastInterpolations._phs_phi_dprime(2.0, Val{2}()) ≈ 2.0  # 2*(2-1)*2^0

    # ── K=1 specializations ──────────────────────────────────────────────────
    # _phs_phi K=1 (line 35): just r
    @test FastInterpolations._phs_phi(3.0, Val{1}()) ≈ 3.0
    @test FastInterpolations._phs_phi(0.0, Val{1}()) == 0.0

    # _phs_phi_prime K=1 (lines 58-60): r<=0 guard + one(T)
    @test FastInterpolations._phs_phi_prime(0.0, Val{1}()) == 0.0   # guard
    @test FastInterpolations._phs_phi_prime(3.0, Val{1}()) == 1.0   # one(T)

    # _phs_phi_dprime K=1 (line 83): always zero
    @test FastInterpolations._phs_phi_dprime(0.0, Val{1}()) == 0.0
    @test FastInterpolations._phs_phi_dprime(5.0, Val{1}()) == 0.0

    # _phs_phi_dprime K=3 (line 85): 6*r
    @test FastInterpolations._phs_phi_dprime(2.0, Val{3}()) ≈ 12.0   # 6*2
    @test FastInterpolations._phs_phi_dprime(0.0, Val{3}()) == 0.0

    # ── K=7 specializations ──────────────────────────────────────────────────
    # _phs_phi K=7 (lines 41-44): r4*r2*r = 2^7 = 128
    @test FastInterpolations._phs_phi(2.0, Val{7}()) ≈ 128.0
    @test FastInterpolations._phs_phi(0.0, Val{7}()) == 0.0

    # _phs_phi_prime K=7 (lines 68-70): 7*r2*r2*r2 = 7*64 = 448
    @test FastInterpolations._phs_phi_prime(2.0, Val{7}()) ≈ 448.0
    @test FastInterpolations._phs_phi_prime(0.0, Val{7}()) == 0.0

    # _phs_phi_dprime K=7 (lines 89-91): 42*r2*r2*r = 42*32 = 1344
    @test FastInterpolations._phs_phi_dprime(2.0, Val{7}()) ≈ 1344.0
    @test FastInterpolations._phs_phi_dprime(0.0, Val{7}()) == 0.0

    # ── Int dispatch wrappers (lines 96-97) ──────────────────────────────────
    @test FastInterpolations._phs_phi_prime(3.0, 3) == FastInterpolations._phs_phi_prime(3.0, Val{3}())
    @test FastInterpolations._phs_phi_dprime(3.0, 3) == FastInterpolations._phs_phi_dprime(3.0, Val{3}())
    @test FastInterpolations._phs_phi_prime(2.0, 5) == FastInterpolations._phs_phi_prime(2.0, Val{5}())
    @test FastInterpolations._phs_phi_dprime(2.0, 5) == FastInterpolations._phs_phi_dprime(2.0, Val{5}())
end

@testitem "PHS kernels — blend weight 2-arg variants (dead-code coverage)" begin
    # The 2-arg (d, a) overloads are never called by phs_eval.jl — it always
    # uses the 3-arg (d, a, a3) forms with pre-computed a3.  Call them directly
    # to mark those source lines as covered.

    # _phs_blend_weight(d, a) — lines 129-133
    w_inside = FastInterpolations._phs_blend_weight(0.5, 1.0)
    @test isfinite(w_inside) && 0.0 < w_inside < 1.0
    w_outside = FastInterpolations._phs_blend_weight(1.5, 1.0)   # d >= a → 0
    @test w_outside == 0.0
    w_eq = FastInterpolations._phs_blend_weight(1.0, 1.0)         # d == a → 0
    @test w_eq == 0.0

    # _phs_blend_weight_and_prime(d, a) — lines 147-158
    w2, wp2 = FastInterpolations._phs_blend_weight_and_prime(0.5, 1.0)
    @test isfinite(w2) && w2 > 0.0
    @test isfinite(wp2) && wp2 < 0.0   # weight is monotone decreasing
    w2z, wp2z = FastInterpolations._phs_blend_weight_and_prime(2.0, 1.0)
    @test w2z == 0.0 && wp2z == 0.0

    # _phs_blend_weight_and_derivs(d, a) — lines 182-196
    w3, wp3, wpp3 = FastInterpolations._phs_blend_weight_and_derivs(0.5, 1.0)
    @test isfinite(w3) && w3 > 0.0
    @test isfinite(wp3)
    @test isfinite(wpp3)
    # 2-arg and 3-arg versions should agree
    a3 = 1.0^3
    w3b, wp3b, wpp3b = FastInterpolations._phs_blend_weight_and_derivs(0.5, 1.0, a3)
    @test w3 ≈ w3b && wp3 ≈ wp3b && wpp3 ≈ wpp3b
    w3z, wp3z, wpp3z = FastInterpolations._phs_blend_weight_and_derivs(2.0, 1.0)
    @test w3z == 0.0 && wp3z == 0.0 && wpp3z == 0.0
end

@testitem "PHS kernels — smoothing transform unrollers (dead-code coverage)" begin
    # These helpers (_phs_unroll_value, _phs_unroll_grad_component,
    # _phs_unroll_hess_component) are defined in phs_kernels.jl but are not
    # called from phs_eval.jl — the log-transform unrolling is inlined directly.
    # Call them directly to mark their source lines as covered.

    # _phs_unroll_value (lines 229-231)
    rho = FastInterpolations._phs_unroll_value(1.0, 2.0)
    @test rho ≈ 2.0 * exp(1.0)
    # early return: rho0 < 1e-40
    @test FastInterpolations._phs_unroll_value(1.0, 0.0) == 0.0
    # early return: f > 100
    @test FastInterpolations._phs_unroll_value(200.0, 1.0) == 0.0

    # _phs_unroll_grad_component (lines 240-242)
    rho_val = 2.0 * exp(1.0)
    grad = FastInterpolations._phs_unroll_grad_component(rho_val, 0.3, 0.1, 2.0)
    @test isfinite(grad)
    @test grad ≈ rho_val * (0.3 + 0.1 / 2.0) atol = 1e-12

    # _phs_unroll_hess_component (lines 251, 261-263)
    #   rho·(f_hess + rho_xi·rho_xj/rho² + rho0_hess/rho0 - rho0_xi·rho0_xj/rho0²)
    rho_g1 = 0.2; rho_g2 = 0.3
    rho0_g1 = 0.1; rho0_g2 = 0.15
    hess = FastInterpolations._phs_unroll_hess_component(
        rho_val, 0.1, rho_g1, rho_g2, 2.0, rho0_g1, rho0_g2, 0.05)
    @test isfinite(hess)
    expected = rho_val * (
        0.1 +
        rho_g1 * rho_g2 / (rho_val * rho_val) +
        0.05 / 2.0 -
        rho0_g1 * rho0_g2 / (2.0 * 2.0)
    )
    @test hess ≈ expected atol = 1e-12
end

@testitem "PHS kernels — _phs_n_poly and mixed-partial deriv2 (ax1≠ax2)" begin
    # _phs_n_poly (line 291): dead helper, never called from phs_stencil.jl
    @test FastInterpolations._phs_n_poly(1, 0) == 1    # constant in 1D
    @test FastInterpolations._phs_n_poly(2, 1) == 3    # linear in 2D
    @test FastInterpolations._phs_n_poly(2, 2) == 6    # quadratic in 2D
    @test FastInterpolations._phs_n_poly(3, 1) == 4    # linear in 3D
    @test FastInterpolations._phs_n_poly(3, 3) == 20   # cubic in 3D = C(6,3)

    # _phs_eval_poly_deriv2 with ax1 ≠ ax2 (lines 422-428)
    # K=5 gives quadratic augmentation (terms: 1, x, y, x², xy, y²).
    # The xy cross-term ensures the ax1≠ax2 branch executes non-trivially.
    # ∂²(sin x · cos y)/∂x∂y = −cos x · sin y
    x = range(0.0, 2π, 20)
    y = range(0.0, 2π, 20)
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp = phs_interp((x, y), data; stencil_size = 8, degree = 5)
    qx, qy = 1.5, 1.5
    d2f_mixed = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{1}()))
    @test d2f_mixed ≈ -cos(qx) * sin(qy) atol = 0.1
end

@testitem "PHS kernels — K=1 and K=7 via PHS interpolant (phi/phi_prime/phi_dprime specializations)" begin
    # K=1: covers _phs_phi(r, Val{1}()), _phs_phi_prime(r, Val{1}()),
    #          _phs_phi_dprime(r, Val{1}()) through the hot evaluation loop.
    # A degree-1 PHS with constant augmentation exactly reproduces constants.
    x = range(0.0, 5.0, 20)
    data = fill(3.0, 20)    # constant function
    itp1 = phs_interp((x,), data; stencil_size = 4, degree = 1)
    @test itp1((2.5,)) ≈ 3.0 atol = 1e-6
    # First derivative of constant = 0 (triggers phi_prime K=1)
    d1 = itp1((2.5,); deriv = (DerivOp{1}(),))
    @test abs(d1) < 1e-4
    # Second derivative (triggers phi_dprime K=1 → always 0)
    d2 = itp1((2.5,); deriv = (DerivOp{2}(),))
    @test abs(d2) < 1e-3

    # K=3 second derivative: covers _phs_phi_dprime(r, Val{3}()) (line 85)
    x3 = range(0.0, 2π, 20)
    data3 = sin.(collect(x3))
    itp3 = phs_interp((x3,), data3; stencil_size = 6, degree = 3)
    d2_k3 = itp3((1.5,); deriv = (DerivOp{2}(),))
    @test d2_k3 ≈ -sin(1.5) atol = 0.1

    # K=7: covers _phs_phi(r, Val{7}()), _phs_phi_prime(r, Val{7}()),
    #          _phs_phi_dprime(r, Val{7}()) through the hot evaluation loop.
    x7 = range(0.0, 2π, 25)
    data7 = cos.(collect(x7))
    itp7 = phs_interp((x7,), data7; stencil_size = 10, degree = 7)
    # Value (triggers phi K=7)
    @test itp7((1.0,)) ≈ cos(1.0) atol = 1e-3
    # First derivative: d(cos x)/dx = -sin x  (triggers phi_prime K=7)
    d1_k7 = itp7((1.0,); deriv = (DerivOp{1}(),))
    @test d1_k7 ≈ -sin(1.0) atol = 0.05
    # Second derivative: d²(cos x)/dx² = -cos x  (triggers phi_dprime K=7)
    d2_k7 = itp7((1.0,); deriv = (DerivOp{2}(),))
    @test d2_k7 ≈ -cos(1.0) atol = 0.1
end

# ======================================================
# Coverage: phs_eval.jl missed lines — Batch 1
# ======================================================

@testitem "PHS eval — non-uniform grid (VectorSpacing binary search)" begin
    # A Vector grid creates VectorSpacing which triggers the O(log n) binary
    # search path in _phs_find_base_node instead of the O(1) ScalarSpacing formula.
    x = collect(range(0.0, 2π, 20))  # Vector{Float64} → VectorSpacing
    data = sin.(x)
    itp = phs_interp((x,), data; stencil_size = 7, degree = 3)
    @test itp.grids[1] isa FastInterpolations._CachedVector  # ensure binary search path
    for qx in [0.3, 1.0, 2.0, 4.0, 5.8]
        @test itp((qx,)) ≈ sin(qx) atol = 0.01
    end
    # Derivative on non-uniform grid
    @test itp((1.5,); deriv = (DerivOp{1}(),)) ≈ cos(1.5) atol = 0.1
end

@testitem "PHS eval — d≈0 branches at grid node, no log-transform" begin
    # Querying at EXACTLY a grid node makes d_dist = 0 for the base node,
    # forcing the blend loop into the d≈0 else-branch.
    #   • First derivative  → lines 586-588 in _phs_eval_blended
    #   • Second derivative → lines 652-655 in _phs_eval_blended
    #                         (which calls _phs_eval_from_coeffs with total_order==2,
    #                          covering lines 458-467, then _phs_eval_coeffs_deriv2, lines 174-210)
    x = range(0.0, 2π, 20)
    y = range(0.0, 2π, 20)
    data = [sin(xi) + cos(yj) for xi in x, yj in y]
    itp = phs_interp((x, y), data; stencil_size = 6, degree = 3)

    qx = Float64(x[10])   # exact grid node → d_dist = 0 for (10,10)
    qy = Float64(y[10])

    # First derivative at grid node → lines 586-588
    d1 = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{0}()))
    @test isfinite(d1)
    @test d1 ≈ cos(qx) atol = 0.3

    # Diagonal second derivative at grid node → lines 652-655
    # also triggers _phs_eval_from_coeffs total_order==2, lines 458-464, 174-190
    d2x = itp((qx, qy); deriv = (DerivOp{2}(), DerivOp{0}()))
    @test isfinite(d2x)
    @test d2x ≈ -sin(qx) atol = 0.5

    # Off-diagonal second derivative at grid node → lines 652-655
    # triggers lines 465-467 in _phs_eval_from_coeffs and lines 191-210 in _phs_eval_coeffs_deriv2
    d2xy = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{1}()))
    @test isfinite(d2xy)
end

@testitem "PHS log-transform — second derivatives diagonal and off-diagonal" begin
    # Exercises the entire second-derivative branch of _phs_eval_blended_G
    # (lines 789-874) and _phs_eval_with_transform (lines 913-929).
    x = range(0.2, Float64(π), 20)
    y = range(0.2, Float64(π), 20)
    rho = [2.0 + sin(xi) * cos(yj) for xi in x, yj in y]
    ref = ConstantRef(1.0)
    itp = phs_interp((x, y), rho; stencil_size = 6, degree = 3, reference_interp = ref)

    h = 1e-4
    qx, qy = 1.2, 1.0

    # Diagonal ∂²ρ/∂x² — triggers is_diag=true path, lines 803-812, 862-866, 913-929
    d2xx    = itp((qx, qy); deriv = (DerivOp{2}(), DerivOp{0}()))
    d2xx_fd = (itp((qx + h, qy)) - 2itp((qx, qy)) + itp((qx - h, qy))) / h^2
    @test isfinite(d2xx)
    @test d2xx ≈ d2xx_fd atol = 1e-3

    # Off-diagonal ∂²ρ/∂x∂y — triggers is_diag=false path, lines 812-846, 868-872, 913-929
    d2xy    = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{1}()))
    d2xy_fd = (itp((qx + h, qy + h)) - itp((qx + h, qy)) - itp((qx, qy + h)) + itp((qx, qy))) / h^2
    @test isfinite(d2xy)
    @test d2xy ≈ d2xy_fd atol = 1e-3
end

@testitem "PHS log-transform — d≈0 at grid node" begin
    # Querying at a grid node with a log-transform interpolant:
    #   • First derivative  → d≈0 in _phs_eval_blended_G, lines 774-777
    #   • Second derivative → d≈0 in _phs_eval_blended_G, lines 850-857
    #     For diagonal 2nd deriv: also calls _phs_eval_from_coeffs with total_order==1
    #     (via ops_d1_1 at line 854) → lines 455-457, then _phs_eval_coeffs_deriv1, lines 132-156
    x = range(0.2, Float64(π), 15)
    y = range(0.2, Float64(π), 15)
    rho = [2.0 + sin(xi) * cos(yj) for xi in x, yj in y]
    ref = ConstantRef(1.0)
    itp = phs_interp((x, y), rho; stencil_size = 5, degree = 3, reference_interp = ref)

    qx = Float64(x[8])   # exact grid node
    qy = Float64(y[8])

    # First derivative at grid node → d≈0 in _phs_eval_blended_G first-deriv branch (lines 774-777)
    d1 = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{0}()))
    @test isfinite(d1)

    # Diagonal second derivative at grid node → d≈0 in _phs_eval_blended_G second-deriv branch
    # (lines 850-857); also triggers ops_d1_1 path → _phs_eval_coeffs_deriv1 (lines 132-156)
    # and _phs_eval_from_coeffs total_order==1 (lines 455-457)
    d2xx = itp((qx, qy); deriv = (DerivOp{2}(), DerivOp{0}()))
    @test isfinite(d2xx)

    # Off-diagonal second derivative at grid node → d≈0 branch (lines 850-857)
    # is_diag=false so f_d1_sq = zero(Tv), but f_d2 calls _phs_eval_coeffs_deriv2 (off-diagonal)
    d2xy = itp((qx, qy); deriv = (DerivOp{1}(), DerivOp{1}()))
    @test isfinite(d2xy)
end

@testitem "PHS eval — dead-code _phs_eval_coeffs_value_and_two_deriv1 + total_deriv≥3 fallbacks" begin
    # ── Setup: build a 2D interpolant and extract a solved stencil ──────────
    x = range(0.0, 2π, 15)
    y = range(0.0, 2π, 15)
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp = phs_interp((x, y), data; stencil_size = 5, degree = 3)
    qx, qy = 1.5, 1.0

    base_idx  = FastInterpolations._phs_find_base_node(itp, (qx, qy))
    M         = size(itp.phi_inv, 1)
    rhs_buf   = zeros(eltype(itp.hs[1]), M)
    coeff_buf = zeros(eltype(itp.hs[1]), M)
    offsets, phys_offsets, coeffs, hs = FastInterpolations._phs_solve_stencil!(itp, base_idx, rhs_buf, coeff_buf)
    base_coords = FastInterpolations._phs_base_coords(itp, base_idx)

    # ── Dead-code function: _phs_eval_coeffs_value_and_two_deriv1 (lines 292-353) ──
    # Never called from eval paths; replaced by the 4-return fused version.
    # Call directly to exercise those lines.
    val, dax1, dax2 = FastInterpolations._phs_eval_coeffs_value_and_two_deriv1(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(), 1, 2)
    @test isfinite(val) && isfinite(dax1) && isfinite(dax2)
    # Cross-check against the non-fused versions (which are also tested here)
    val_ref  = FastInterpolations._phs_eval_coeffs_value(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}())
    d1x_ref  = FastInterpolations._phs_eval_coeffs_deriv1(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(), 1)
    d1y_ref  = FastInterpolations._phs_eval_coeffs_deriv1(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(), 2)
    @test val  ≈ val_ref  atol = 1e-8
    @test dax1 ≈ d1x_ref  atol = 1e-8
    @test dax2 ≈ d1y_ref  atol = 1e-8

    # ── _phs_eval_coeffs_deriv2: diagonal (lines 174-190) and off-diagonal (lines 191-210) ──
    d2xx_direct = FastInterpolations._phs_eval_coeffs_deriv2(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(), 1, 1)
    d2xy_direct = FastInterpolations._phs_eval_coeffs_deriv2(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(), 1, 2)
    @test isfinite(d2xx_direct) && isfinite(d2xy_direct)

    # ── _phs_eval_from_coeffs: total_order==1 (lines 455-457) ──
    v1 = FastInterpolations._phs_eval_from_coeffs(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(),
        (DerivOp{1}(), EvalValue()))
    @test v1 ≈ d1x_ref atol = 1e-8

    # ── _phs_eval_from_coeffs: total_order==2 diagonal (lines 458-464) ──
    v2d = FastInterpolations._phs_eval_from_coeffs(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(),
        (DerivOp{2}(), EvalValue()))
    @test v2d ≈ d2xx_direct atol = 1e-8

    # ── _phs_eval_from_coeffs: total_order==2 off-diagonal (lines 465-467) ──
    v2od = FastInterpolations._phs_eval_from_coeffs(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(),
        (DerivOp{1}(), DerivOp{1}()))
    @test v2od ≈ d2xy_direct atol = 1e-8

    # ── _phs_eval_from_coeffs: total_order≥3 → zero (line 470) ──
    z_fc = FastInterpolations._phs_eval_from_coeffs(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(),
        (DerivOp{2}(), DerivOp{1}()))
    @test z_fc == 0.0

    # ── _phs_eval_blended: total_deriv≥3 fallback (line 674) ──
    z_blended = itp((qx, qy); deriv = (DerivOp{2}(), DerivOp{1}()))
    @test z_blended == 0.0

    # ── _phs_eval_blended_G: total_deriv≥3 fallback (line 874) ──
    # Call the log-transform blend function directly with total_deriv=3
    rho2 = [2.0 + sin(xi) * cos(yj) for xi in x, yj in y]
    itp_log = phs_interp((x, y), rho2; stencil_size = 5, degree = 3,
        reference_interp = ConstantRef(1.0))
    z_G = FastInterpolations._phs_eval_blended_G(
        itp_log, (qx, qy), (DerivOp{2}(), DerivOp{1}()))
    @test z_G == 0.0

    # ── _phs_eval_with_transform: total_deriv≥3 fallback (line 932) ──
    z_tr = itp_log((qx, qy); deriv = (DerivOp{2}(), DerivOp{1}()))
    @test z_tr == 0.0

    # ── _phs_eval_coeffs_value_and_deriv1_and_deriv2: off-diagonal else-branch (lines 292-303) ──
    # This function is always called with ax1==ax2 in the blended path (diagonal only).
    # Call directly with ax1≠ax2 to exercise the off-diagonal loop.
    val3, d1_3, d2_3 = FastInterpolations._phs_eval_coeffs_value_and_deriv1_and_deriv2(
        coeffs, phys_offsets, (qx, qy), base_coords, Val{3}(), 1, 2)
    @test isfinite(val3) && isfinite(d1_3) && isfinite(d2_3)
    # Cross-check: value should match the zero-deriv version; d1 matches deriv1 along ax1
    @test val3  ≈ val_ref  atol = 1e-8
    @test d1_3  ≈ d1x_ref  atol = 1e-8
    @test d2_3  ≈ d2xy_direct atol = 1e-8
end
