# ND scalar one-shot on mixed-precision / heterogeneous-eltype grids.
#
# After the raw-grid migration every method must still evaluate on axis tuples
# whose eltypes differ (e.g. `Float32 × Float64`) and under AD-wrt-grid (a Dual
# axis beside a Float axis). Linear's `_multilinear_sum` constrains `inv_hs` to a
# homogeneous tuple, so genuinely-mixed-float grids throw `MethodError` until that
# signature is relaxed. Cubic/quadratic/hermite promote spacings via
# `_compute_all_local_params`, and constant is a selection kernel — all handled.

@testitem "ND one-shot mixed-precision grids — value matches all-Float64" begin
    fq(a, b) = 2.0a + 3.0b - 0.5a * b + 1.0
    q = (1.5, 2.5)
    # Genuinely mixed-eltype axes: a Float32 axis keeps a Float32 `inv_h` that
    # cannot collapse to the other axis' Float64 (unlike Int, whose `inv_h` floats).
    mixes = [
        ("Int × Float64", [0, 1, 2, 3, 4], Float64.(0:3)),          # control (both inv_h Float64)
        ("Float32 × Float64", Float32.(0:4), Float64.(0:3)),
        ("Float64 × Float32", Float64.(0:4), Float32.(0:3)),
        ("Int × Float32", [0, 1, 2, 3, 4], Float32.(0:3)),
        ("Float64Range × IntVec", 0.0:4.0, [0, 1, 2, 3]),           # control (both inv_h Float64)
    ]
    for (name, gx, gy) in mixes
        xc = collect(gx)
        yc = collect(gy)
        data = [fq(a, b) for a in xc, b in yc]
        ref_lin = linear_interp((Float64.(xc), Float64.(yc)), data, q)
        ref_con = constant_interp((Float64.(xc), Float64.(yc)), data, q)
        @testset "linear $name" begin
            @test linear_interp((gx, gy), data, q) ≈ ref_lin rtol = 1.0e-5
        end
        @testset "constant $name" begin
            @test constant_interp((gx, gy), data, q) === ref_con
        end
    end
end

@testitem "Linear ND one-shot — AD wrt grid nodes (mixed Dual/Float axis)" begin
    using ForwardDiff
    x = [0.0, 1, 2, 3, 4]
    y = [0.0, 1, 2, 3]
    q = (1.5, 2.5)
    data = [2.0a + 3.0b - 0.5a * b for a in x, b in y]   # bilinear → linear is exact
    # Central-difference reference for ∂value/∂(x-nodes); the all-Float64 path works today.
    fd = map(eachindex(x)) do i
        xp = copy(x)
        xm = copy(x)
        xp[i] += 1.0e-6
        xm[i] -= 1.0e-6
        (linear_interp((xp, y), data, q) - linear_interp((xm, y), data, q)) / 2.0e-6
    end
    @testset "∂/∂x-nodes (x Dual, y Float64)" begin
        g = ForwardDiff.gradient(xx -> linear_interp((xx, y), data, q), x)
        @test g ≈ fd atol = 1.0e-6
    end
    @testset "homogeneous control: ∂/∂[x; y] (both axes Dual)" begin
        g = ForwardDiff.gradient(p -> linear_interp((p[1:5], p[6:9]), data, q), [x; y])
        @test all(isfinite, g)
    end
end

@testitem "Cubic/Quadratic/Hermite ND one-shot — AD wrt grid nodes" begin
    using ForwardDiff
    using FastInterpolations: HermitePartials
    x = [0.0, 1, 2, 3, 4, 5]
    y = [0.0, 1, 2, 3, 4]
    q = (3.4, 2.6)
    # Central-difference reference for ∂value/∂(x-nodes); the all-Float64 path works today.
    fdx(f) = map(eachindex(x)) do i
        xp = copy(x)
        xm = copy(x)
        xp[i] += 1.0e-6
        xm[i] -= 1.0e-6
        (f((xp, y)) - f((xm, y))) / 2.0e-6
    end
    data = [sin(0.7a) + cos(0.5b) for a in x, b in y]
    @testset "cubic ∂/∂x-nodes finite & matches FD" begin
        g = ForwardDiff.gradient(xx -> cubic_interp((xx, y), data, q), x)
        @test all(isfinite, g)
        @test g ≈ fdx(gg -> cubic_interp(gg, data, q)) atol = 1.0e-3
    end
    @testset "quadratic ∂/∂x-nodes finite & matches FD" begin
        g = ForwardDiff.gradient(xx -> quadratic_interp((xx, y), data, q), x)
        @test all(isfinite, g)
        @test g ≈ fdx(gg -> quadratic_interp(gg, data, q)) atol = 1.0e-3
    end
    @testset "hermite ∂/∂x-nodes finite & matches FD" begin
        p = HermitePartials(
            (1, 0) => [cos(1.0a) * cos(1.0b) for a in x, b in y],
            (0, 1) => [-sin(1.0a) * sin(1.0b) for a in x, b in y],
            (1, 1) => [-cos(1.0a) * sin(1.0b) for a in x, b in y],
        )
        dH = [sin(1.0a) * cos(1.0b) for a in x, b in y]
        g = ForwardDiff.gradient(xx -> hermite_interp((xx, y), dH, p, q), x)
        @test all(isfinite, g)
        @test g ≈ fdx(gg -> hermite_interp(gg, dH, p, q)) atol = 1.0e-3
    end
end
