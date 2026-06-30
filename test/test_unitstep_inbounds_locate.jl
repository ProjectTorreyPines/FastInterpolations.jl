# Behavior pin for the canonical `_UnitStep` + `InBounds` ND locate fast path.
#
# This is a perf refactor (the correct results already exist via the generic
# `_locate_cell`), so these are characterization tests: green on the pre-refactor
# baseline and green after. Two assertions carry real signal for the refactor:
#   1. value/deriv match an INDEPENDENT manual multilinear reference (catches any
#      idx/α/stencil mistake in the fast path), and
#   2. the InBounds path is BIT-IDENTICAL (`===`) to the NoExtrap path on the same
#      `_UnitStep` grid for in-domain queries — this fails the moment the fast
#      path's arithmetic deviates by even one ULP from the generic path.

@testitem "UnitStep InBounds locate — 2D matches reference, bit-identical to NoExtrap" begin
    using FastInterpolations: InBounds, DerivOp, GridIdx

    # Independent bilinear reference on unit-step axes (node index == position).
    function ref_bilin(axx, axy, data, qx, qy)
        lox, loy = first(axx), first(axy)
        nx, ny = length(axx), length(axy)
        ix = clamp(floor(Int, qx - lox) + 1, 1, nx - 1); αx = qx - (lox + (ix - 1))
        iy = clamp(floor(Int, qy - loy) + 1, 1, ny - 1); αy = qy - (loy + (iy - 1))
        v0 = (1 - αx) * data[ix, iy] + αx * data[ix + 1, iy]
        v1 = (1 - αx) * data[ix, iy + 1] + αx * data[ix + 1, iy + 1]
        return (1 - αy) * v0 + αy * v1
    end

    axx, axy = 1:7, 2:9                      # two different 1-step lo's (1 and 2)
    data = [sin(0.3i) + cos(0.2j) + 0.01i * j for i in 1:length(axx), j in 1:length(axy)]
    inb = linear_interp((axx, axy), data; extrap = (InBounds(), InBounds()))
    nox = linear_interp((axx, axy), data)    # default NoExtrap, same _UnitStep grids

    qs = ((1.0, 2.0), (1.25, 4.75), (3.4, 6.2), (6.999, 8.5), (7.0, 9.0))  # incl. q==hi
    @testset "value: matches reference and === NoExtrap" begin
        for q in qs
            @test inb(q) ≈ ref_bilin(axx, axy, data, q...)
            @test inb(q) === nox(q)          # bit-identical to the generic path
        end
    end

    @testset "right-boundary q==hi returns the top corner" begin
        @test inb((7.0, 9.0)) === data[end, end]
    end

    @testset "first derivatives match reference slopes (deriv stays generic)" begin
        for q in ((1.25, 4.75), (3.4, 6.2))
            qx, qy = q
            ix = clamp(floor(Int, qx - 1) + 1, 1, length(axx) - 1)
            iy = clamp(floor(Int, qy - 2) + 1, 1, length(axy) - 1)
            αx = qx - ix; αy = qy - (1 + iy)         # node positions: x→ix, y→iy+1
            dvdx = (1 - αy) * (data[ix + 1, iy] - data[ix, iy]) +
                αy * (data[ix + 1, iy + 1] - data[ix, iy + 1])
            dvdy = (1 - αx) * (data[ix, iy + 1] - data[ix, iy]) +
                αx * (data[ix + 1, iy + 1] - data[ix + 1, iy])
            @test inb(q; deriv = DerivOp(1, 0)) ≈ dvdx
            @test inb(q; deriv = DerivOp(0, 1)) ≈ dvdy
            @test inb(q; deriv = DerivOp(1, 0)) === nox(q; deriv = DerivOp(1, 0))
            @test inb(q; deriv = DerivOp(0, 1)) === nox(q; deriv = DerivOp(0, 1))
        end
    end

    @testset "GridIdx queries === NoExtrap" begin
        @test inb((GridIdx(7), 8.5)) === nox((GridIdx(7), 8.5))
        @test inb((3.25, GridIdx(8))) === nox((3.25, GridIdx(8)))
        @test inb((GridIdx(1), GridIdx(1))) === data[1, 1]
    end

    @testset "looped scalar eval is non-allocating" begin
        # Loop barrier matches real looped-scalar use (imresize); the per-call
        # kwarg overhead a single `@allocated` would catch is elided in the loop.
        function fill_loop!(out, itp, qxs, qys)
            @inbounds for i in eachindex(qxs, qys)
                out[i] = itp(qxs[i], qys[i])
            end
            return out
        end
        qxs = collect(range(1.05, 6.95; length = 64))
        qys = collect(range(2.05, 8.95; length = 64))
        out = similar(qxs)
        fill_loop!(out, inb, qxs, qys)               # warmup/compile
        @test (@allocated fill_loop!(out, inb, qxs, qys)) == 0
    end
end

@testitem "UnitStep InBounds locate — 3D matches reference, bit-identical to NoExtrap" begin
    using FastInterpolations: InBounds

    function ref_trilin(ax, data, qx, qy, qz)
        lo = map(first, ax); n = map(length, ax)
        ix = clamp(floor(Int, qx - lo[1]) + 1, 1, n[1] - 1); αx = qx - (lo[1] + (ix - 1))
        iy = clamp(floor(Int, qy - lo[2]) + 1, 1, n[2] - 1); αy = qy - (lo[2] + (iy - 1))
        iz = clamp(floor(Int, qz - lo[3]) + 1, 1, n[3] - 1); αz = qz - (lo[3] + (iz - 1))
        c(a, b, t) = (1 - t) * a + t * b
        v00 = c(data[ix, iy, iz], data[ix + 1, iy, iz], αx)
        v10 = c(data[ix, iy + 1, iz], data[ix + 1, iy + 1, iz], αx)
        v01 = c(data[ix, iy, iz + 1], data[ix + 1, iy, iz + 1], αx)
        v11 = c(data[ix, iy + 1, iz + 1], data[ix + 1, iy + 1, iz + 1], αx)
        return c(c(v00, v10, αy), c(v01, v11, αy), αz)
    end

    ax = (1:5, 1:6, 1:7)
    data = [0.1i + 0.2j + 0.3k + 0.01i * j * k for i in 1:5, j in 1:6, k in 1:7]
    inb = linear_interp(ax, data; extrap = (InBounds(), InBounds(), InBounds()))
    nox = linear_interp(ax, data)

    for q in ((1.0, 1.0, 1.0), (2.3, 4.1, 5.9), (4.999, 5.5, 6.001), (5.0, 6.0, 7.0))
        @test inb(q) ≈ ref_trilin(ax, data, q...)
        @test inb(q) === nox(q)
    end
end

@testitem "per-axis extrap: lean search applies per axis (mixed InBounds/Clamp)" begin
    # The lean InBounds search is threaded per-axis through `_search_axis_adaptive`, so a
    # mixed `(InBounds, ClampExtrap)` interpolant leans ONLY the InBounds axis while the
    # Clamp axis keeps its domain handling. In-domain queries must stay bit-identical.
    using FastInterpolations: InBounds, ClampExtrap

    data = [0.1i + 0.3j + 0.01i * j for i in 1:7, j in 1:8]
    grids = (1:7, 2:9)
    mixed = linear_interp(grids, data; extrap = (InBounds(), ClampExtrap()))
    allclamp = linear_interp(grids, data; extrap = (ClampExtrap(), ClampExtrap()))
    nox = linear_interp(grids, data)

    @testset "in-domain: InBounds (lean) axis === domain-checked paths" begin
        for q in ((1.5, 3.5), (4.25, 6.75), (7.0, 9.0))
            @test mixed(q) === nox(q)
            @test mixed(q) === allclamp(q)
        end
    end

    @testset "Clamp axis still clamps while InBounds axis stays in-domain" begin
        # y above the grid → Clamp axis pins to last; x (InBounds) in-domain, no check.
        @test mixed((3.5, 100.0)) === allclamp((3.5, 100.0))
    end
end
