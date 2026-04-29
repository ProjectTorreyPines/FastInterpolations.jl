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
