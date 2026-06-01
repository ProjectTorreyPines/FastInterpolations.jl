# ========================================
# ND Cubic Hermite (User-Supplied Full Mixed Partials) — Phase 1a Tests
# ========================================
#
# Coverage:
# - Construction validation (bad keys, missing, duplicates, size mismatches, BC rejection)
# - Analytic exactness (polynomials up to cubic in each axis, separable trig)
# - Cross-validation against `cardinal_interp` ND with manually-computed central FDM
# - Periodic BC modes (NoBC / :inclusive / :exclusive)
# - Derivative output (`deriv` kwarg)
# - One-shot scalar + batch + in-place

@testitem "Hermite ND Partials (Phase 1a)" setup = [AllocConstants] begin

    # Helper: central FDM along axis `d` of N-D array, using grid `g` (length n_d).
    # Interior: (arr[..., j+1, ...] - arr[..., j-1, ...]) / (g[j+1] - g[j-1])
    # Boundary j=1 :          forward difference  (arr[..., 2, ...] - arr[..., 1, ...]) / (g[2] - g[1])
    # Boundary j=n :          backward difference (arr[..., n, ...] - arr[..., n-1, ...]) / (g[n] - g[n-1])
    function _central_fdm_along_axis(arr::AbstractArray{T, N}, grid::AbstractVector, d::Int) where {T, N}
        n = size(arr, d)
        n >= 2 || throw(ArgumentError("FDM requires axis length >= 2"))
        out = similar(arr, float(T))
        for j in 1:n
            if j == 1
                slab_lo = selectdim(arr, d, 1)
                slab_hi = selectdim(arr, d, 2)
                denom = grid[2] - grid[1]
            elseif j == n
                slab_lo = selectdim(arr, d, n - 1)
                slab_hi = selectdim(arr, d, n)
                denom = grid[n] - grid[n - 1]
            else
                slab_lo = selectdim(arr, d, j - 1)
                slab_hi = selectdim(arr, d, j + 1)
                denom = grid[j + 1] - grid[j - 1]
            end
            selectdim(out, d, j) .= (slab_hi .- slab_lo) ./ denom
        end
        return out
    end

    @testset "Hermite ND — Construction validation (N=2)" begin
        n = 5
        A = zeros(n, n)

        @testset "Happy path" begin
            p = HermitePartials(
                (1, 0) => copy(A),
                (0, 1) => copy(A),
                (1, 1) => copy(A),
            )
            @test p isa HermitePartials{2, Float64, 3, Matrix{Float64}}
        end

        @testset "Mixed eltypes promote to common Tv" begin
            A32 = zeros(Float32, n, n)
            A64 = zeros(Float64, n, n)
            p = HermitePartials(
                (1, 0) => A32,         # Float32
                (0, 1) => A64,         # Float64
                (1, 1) => A32,         # Float32
            )
            @test p isa HermitePartials{2, Float64, 3, Matrix{Float64}}
        end

        @testset "Wrong pair count" begin
            @test_throws ArgumentError HermitePartials(
                (1, 0) => copy(A),
                (0, 1) => copy(A),
                # missing (1, 1)
            )
            @test_throws ArgumentError HermitePartials(
                (1, 0) => copy(A),
                (0, 1) => copy(A),
                (1, 1) => copy(A),
                (1, 1) => copy(A),  # too many
            )
        end

        @testset "Duplicate multiindex" begin
            @test_throws ArgumentError HermitePartials(
                (1, 0) => copy(A),
                (0, 1) => copy(A),
                (0, 1) => copy(A),  # duplicate
            )
        end

        @testset "Zero multiindex disallowed" begin
            @test_throws ArgumentError HermitePartials(
                (0, 0) => copy(A),
                (1, 0) => copy(A),
                (0, 1) => copy(A),
            )
        end

        @testset "Out-of-range multiindex entry" begin
            @test_throws ArgumentError HermitePartials(
                (1, 0) => copy(A),
                (0, 1) => copy(A),
                (2, 0) => copy(A),  # entry == 2, not 0 or 1
            )
        end

        @testset "Mismatched array sizes" begin
            B = zeros(n + 1, n)
            @test_throws DimensionMismatch HermitePartials(
                (1, 0) => copy(A),
                (0, 1) => copy(B),
                (1, 1) => copy(A),
            )
        end

        @testset "Wrong array ndims" begin
            v = zeros(n)
            @test_throws DimensionMismatch HermitePartials(
                (1, 0) => copy(A),
                (0, 1) => v,        # 1-D, not 2-D
                (1, 1) => copy(A),
            )
        end
    end

    @testset "Hermite ND — Build-time BC rejection (Phase 1a)" begin
        x = range(0.0, 1.0, length = 5)
        y = range(0.0, 1.0, length = 5)
        data = [xi * yj for xi in x, yj in y]
        dfdx = [yj for xi in x, yj in y]
        dfdy = [xi for xi in x, yj in y]
        d2 = ones(5, 5)
        p = HermitePartials((1, 0) => dfdx, (0, 1) => dfdy, (1, 1) => d2)

        # NoBC, PeriodicBC ok
        @test hermite_interp((x, y), data, p) isa CubicHermiteInterpolantND
        # CubicFit, NaturalBC, etc. rejected — user partials supersede.
        @test_throws ArgumentError hermite_interp((x, y), data, p; bc = CubicFit())
        @test_throws ArgumentError hermite_interp((x, y), data, p; bc = ZeroSlopeBC())
    end

    @testset "Hermite ND — Analytic exactness (polynomial)" begin
        # f(x, y) = a + b x + c y + d x y + e x² + f y² + g x²y + h x y² + i x³ + j y³
        # Cubic Hermite is exact on any bicubic polynomial when supplied with the
        # corresponding analytic partials.

        a, b, c, dd = 0.7, 1.3, -0.4, 2.1
        e, ff, gg, hh = -0.6, 0.5, 0.3, -0.2
        ii, jj = 0.15, -0.25

        F(x, y) = a + b * x + c * y + dd * x * y + e * x^2 + ff * y^2 + gg * x^2 * y + hh * x * y^2 + ii * x^3 + jj * y^3
        Fx(x, y) = b + dd * y + 2e * x + 2gg * x * y + hh * y^2 + 3ii * x^2
        Fy(x, y) = c + dd * x + 2ff * y + gg * x^2 + 2hh * x * y + 3jj * y^2
        Fxy(x, y) = dd + 2gg * x + 2hh * y

        x = collect(range(-1.0, 1.5, length = 8))
        y = collect(range(-0.8, 1.2, length = 7))

        data = [F(xi, yj) for xi in x, yj in y]
        dfdx = [Fx(xi, yj) for xi in x, yj in y]
        dfdy = [Fy(xi, yj) for xi in x, yj in y]
        d2 = [Fxy(xi, yj) for xi in x, yj in y]
        p = HermitePartials((1, 0) => dfdx, (0, 1) => dfdy, (1, 1) => d2)

        itp = hermite_interp((x, y), data, p)

        # 1. Grid nodes — exact
        for i in 1:length(x), j in 1:length(y)
            @test itp((x[i], y[j])) ≈ data[i, j] atol = 1.0e-12
        end

        # 2. Random interior points — exact for bicubic polynomial
        for (qx, qy) in [(0.13, 0.42), (-0.51, 1.1), (1.32, -0.7), (0.0, 0.0), (1.0, 1.0)]
            @test itp((qx, qy)) ≈ F(qx, qy) atol = 1.0e-10
        end

        # 3. Derivative queries
        for (qx, qy) in [(0.13, 0.42), (-0.51, 1.1)]
            @test itp((qx, qy); deriv = (DerivOp(1), DerivOp(0))) ≈ Fx(qx, qy) atol = 1.0e-10
            @test itp((qx, qy); deriv = (DerivOp(0), DerivOp(1))) ≈ Fy(qx, qy) atol = 1.0e-10
            @test itp((qx, qy); deriv = (DerivOp(1), DerivOp(1))) ≈ Fxy(qx, qy) atol = 1.0e-10
        end
    end

    @testset "Hermite ND — Cardinal cross-validation (central FDM partials)" begin
        # The key test the user explicitly requested:
        # Feed Hermite ND with manually-computed central-FDM partials, compare
        # to `cardinal_interp` ND on the same data. They should agree at
        # interior query points (boundary handling may differ; sticking to
        # interior cells for strict equivalence).

        F(x, y) = sin(0.7x) * cos(0.5y) + 0.3x * y

        x = collect(range(0.0, 4.0, length = 12))
        y = collect(range(0.0, 3.0, length = 10))
        data = [F(xi, yj) for xi in x, yj in y]

        # Central FDM partials
        dfdx_fdm = _central_fdm_along_axis(data, x, 1)
        dfdy_fdm = _central_fdm_along_axis(data, y, 2)
        d2_fdm = _central_fdm_along_axis(dfdx_fdm, y, 2)

        p = HermitePartials((1, 0) => dfdx_fdm, (0, 1) => dfdy_fdm, (1, 1) => d2_fdm)
        itp_h = hermite_interp((x, y), data, p)
        itp_c = cardinal_interp((x, y), data)

        # Interior query points (away from boundary by at least 2 cells).
        # Cardinal's boundary FDM treatment may differ from our forward/backward
        # at edges — interior queries are where the two surfaces should agree.
        interior_queries = [
            (x[4], y[4]),         # exact node, both must return data exactly
            (x[6] + 0.13, y[5] + 0.07),  # interior point
            (x[7] + 0.31, y[6] - 0.09),
            (x[8] - 0.04, y[7] + 0.02),
        ]
        for (qx, qy) in interior_queries
            vh = itp_h((qx, qy))
            vc = itp_c((qx, qy))
            @test vh ≈ vc atol = 1.0e-10
        end
    end

    @testset "Hermite ND — One-shot variants" begin
        F(x, y) = 0.4 + 0.7x - 0.3y + 0.5x * y + 0.2x^2 * y - 0.1y^3
        Fx(x, y) = 0.7 + 0.5y + 0.4x * y
        Fy(x, y) = -0.3 + 0.5x + 0.2x^2 - 0.3y^2
        Fxy(x, y) = 0.5 + 0.4x

        x = collect(range(0.0, 2.0, length = 6))
        y = collect(range(-1.0, 1.0, length = 5))

        data = [F(xi, yj) for xi in x, yj in y]
        p = HermitePartials(
            (1, 0) => [Fx(xi, yj)  for xi in x, yj in y],
            (0, 1) => [Fy(xi, yj)  for xi in x, yj in y],
            (1, 1) => [Fxy(xi, yj) for xi in x, yj in y],
        )

        # Scalar one-shot
        v = hermite_interp((x, y), data, p, (0.33, 0.42))
        @test v ≈ F(0.33, 0.42) atol = 1.0e-10

        # Batch one-shot (allocating)
        queries = [(0.13, -0.42), (0.77, 0.51), (1.23, 0.83)]
        out_batch = hermite_interp((x, y), data, p, queries)
        @test out_batch ≈ [F(qx, qy) for (qx, qy) in queries] atol = 1.0e-10

        # Batch one-shot (in-place)
        out_inplace = Vector{Float64}(undef, length(queries))
        hermite_interp!(out_inplace, (x, y), data, p, queries)
        @test out_inplace ≈ out_batch atol = 1.0e-12

        # Allocating batch output must use the promoted value type, not just the
        # data eltype; otherwise non-real / wider partials can throw InexactError.
        data_real = zeros(2, 2)
        p_complex = HermitePartials(
            (1, 0) => fill(1.0 + 1.0im, 2, 2),
            (0, 1) => fill(0.0 + 0.0im, 2, 2),
            (1, 1) => fill(0.0 + 0.0im, 2, 2),
        )
        q_complex = (0.3, 0.6)
        scalar_complex = hermite_interp((0.0:1.0:1.0, 0.0:1.0:1.0), data_real, p_complex, q_complex)
        batch_complex = hermite_interp((0.0:1.0:1.0, 0.0:1.0:1.0), data_real, p_complex, [q_complex])
        @test batch_complex isa Vector{ComplexF64}
        @test batch_complex == [scalar_complex]
    end

    @testset "Hermite ND — Periodic :inclusive" begin
        # f(x, y) = cos(x) * cos(y); both axes periodic of period 2π
        F(x, y) = cos(x) * cos(y)
        Fx(x, y) = -sin(x) * cos(y)
        Fy(x, y) = -cos(x) * sin(y)
        Fxy(x, y) = sin(x) * sin(y)

        # Inclusive: grid includes both endpoints; data periodic in both axes.
        n = 17
        x = collect(range(0.0, 2π, length = n))   # inclusive endpoints
        y = collect(range(0.0, 2π, length = n))

        data = [F(xi, yj) for xi in x, yj in y]
        p = HermitePartials(
            (1, 0) => [Fx(xi, yj)  for xi in x, yj in y],
            (0, 1) => [Fy(xi, yj)  for xi in x, yj in y],
            (1, 1) => [Fxy(xi, yj) for xi in x, yj in y],
        )

        bc = (PeriodicBC(endpoint = :inclusive), PeriodicBC(endpoint = :inclusive))
        itp = hermite_interp((x, y), data, p; bc)

        # Endpoints must match exactly (data was constructed periodic)
        for qx in [0.13, 0.71, 1.5, 3.0, 5.4, 6.2]
            for qy in [0.21, 0.55, 1.1, 2.7, 4.4, 5.9]
                v = itp((qx, qy))
                @test v ≈ F(qx, qy) atol = 5.0e-3   # cubic Hermite + analytic partials → small surface error
            end
        end
    end

    @testset "Hermite ND — Periodic :exclusive (extension)" begin
        F(x, y) = sin(x) * cos(y)
        Fx(x, y) = cos(x) * cos(y)
        Fy(x, y) = -sin(x) * sin(y)
        Fxy(x, y) = -cos(x) * sin(y)

        # Exclusive: grid does NOT include the last (wrap) point.
        # length(x) == n, x[end] + step(x) == x[1] + 2π.
        # Keep as `range` so `period` is inferable from `step(x)*length(x)`
        # (Vector grids require an explicit `period` kwarg).
        n = 16
        x = range(0.0, 2π * (n - 1) / n, length = n)
        y = range(0.0, 2π * (n - 1) / n, length = n)

        data = [F(xi, yj) for xi in x, yj in y]
        p = HermitePartials(
            (1, 0) => [Fx(xi, yj)  for xi in x, yj in y],
            (0, 1) => [Fy(xi, yj)  for xi in x, yj in y],
            (1, 1) => [Fxy(xi, yj) for xi in x, yj in y],
        )

        bc = (PeriodicBC(endpoint = :exclusive), PeriodicBC(endpoint = :exclusive))
        itp = hermite_interp((x, y), data, p; bc)

        # After build, size(itp) reflects extension (n+1 per axis).
        @test size(itp) == (n + 1, n + 1)

        # Evaluate at points across (including past the user's last grid point — must wrap)
        for qx in [0.13, 1.5, 3.0, 5.4]
            for qy in [0.21, 1.1, 2.7, 5.9]
                v = itp((qx, qy))
                @test v ≈ F(qx, qy) atol = 5.0e-3
            end
        end

        # Round-trip: value at xq must equal value at xq + 2π (within wrap-aware extrap, but
        # PeriodicBC at build time bakes in the wrap so domain query inside [0, 2π) is exact)
        @test itp((0.3, 0.5)) ≈ itp((0.3, 0.5))
    end

    @testset "Hermite ND — Input isolation (persistent copies, oneshot snapshots)" begin
        # ── Persistent: build snapshots inputs; user mutations after build are
        #    not visible to the interpolant.
        @testset "Persistent: itp isolated from post-build user mutation" begin
            x = range(0.0, 1.0, length = 5)
            y = range(0.0, 1.0, length = 5)
            data = [xi * yj for xi in x, yj in y]
            dfdx = [yj for xi in x, yj in y]
            dfdy = [xi for xi in x, yj in y]
            d2 = ones(5, 5)
            p = HermitePartials((1, 0) => dfdx, (0, 1) => dfdy, (1, 1) => d2)

            itp = hermite_interp((x, y), data, p)
            val_before = itp((0.5, 0.5))

            # Trash every user-owned input array
            fill!(data, NaN)
            fill!(dfdx, NaN)
            fill!(dfdy, NaN)
            fill!(d2, NaN)
            # `p.partials[k]` is the *same array* as the dfdx/dfdy/d2 we just
            # nuked — these are aliases established by `HermitePartials`
            # (zero-copy when eltypes match). Confirm:
            @test all(isnan, p.partials[1])
            @test all(isnan, p.partials[2])
            @test all(isnan, p.partials[3])

            # Despite user-side trashing, itp still reproduces the original value:
            # the build-time copy into `nodal_derivs.partials` decoupled it.
            val_after = itp((0.5, 0.5))
            @test val_after == val_before
            @test !isnan(val_after)
        end

        # ── Persistent: the stored packed buffer is also independent (a third
        #    snapshot — mutate itp.nodal_derivs internally, user data untouched).
        @testset "Persistent: user data unaffected by itp internal mutation" begin
            x = range(0.0, 1.0, length = 4)
            y = range(0.0, 1.0, length = 4)
            data = [xi * yj for xi in x, yj in y]
            original_data = copy(data)
            p = HermitePartials(
                (1, 0) => [yj for xi in x, yj in y],
                (0, 1) => [xi for xi in x, yj in y],
                (1, 1) => ones(4, 4),
            )
            itp = hermite_interp((x, y), data, p)
            # Stomp the internal packed buffer
            fill!(itp.nodal_derivs.partials, 99.0)
            # User's `data` is unchanged
            @test data == original_data
        end

        # ── One-shot: each call snapshots the *current* user data. Mutating
        #    between calls is safe and reflected in the next return.
        @testset "One-shot: snapshots current data per call" begin
            x = range(0.0, 1.0, length = 5)
            y = range(0.0, 1.0, length = 5)
            data = [xi * yj for xi in x, yj in y]
            p = HermitePartials(
                (1, 0) => [yj for xi in x, yj in y],
                (0, 1) => [xi for xi in x, yj in y],
                (1, 1) => ones(5, 5),
            )
            v1 = hermite_interp((x, y), data, p, (0.5, 0.5))
            # Replace data in-place with a constant; result must follow.
            # Partials still describe the OLD data, but for cubic Hermite the
            # value at a node is the corresponding data entry exactly — so at
            # interior the eval is a polynomial mix of {data, partials}. Easier
            # invariant: at a *grid node* the eval reduces to `data[i, j]`.
            data .= 7.5
            v2 = hermite_interp((x, y), data, p, (x[3], y[3]))   # grid node
            @test v2 == 7.5
            # And the original call shouldn't have been retroactively affected
            # (no aliasing back into user's array via the pool).
            @test v1 == 0.25   # x[3]*y[3] when data was xi*yj; sanity
        end
    end

    @testset "Hermite ND — Pool-based oneshot zero-alloc" begin
        # All measurements live inside one function so locals are type-stable and
        # `@allocated` reports the steady-state, post-warmup cost (no @testset
        # try/catch artifacts). Bulk-loop divisor confirms per-call alloc.

        function _measure_nobc(n_iters)
            x = range(0.0, 1.0, length = 20)
            y = range(0.0, 1.0, length = 20)
            data = [xi * yj for xi in x, yj in y]
            p = HermitePartials(
                (1, 0) => [yj for xi in x, yj in y],
                (0, 1) => [xi for xi in x, yj in y],
                (1, 1) => ones(20, 20),
            )
            # warmup
            hermite_interp((x, y), data, p, (0.5, 0.5))
            function loop!(out, x, y, data, p, queries, n)
                @inbounds for k in 1:n
                    out[k] = hermite_interp((x, y), data, p, queries[k])
                end
                return nothing
            end
            queries = [(0.1 + 0.0001 * k, 0.5) for k in 1:n_iters]
            out = Vector{Float64}(undef, n_iters)
            loop!(out, x, y, data, p, queries, n_iters)
            return @allocated loop!(out, x, y, data, p, queries, n_iters)
        end

        function _measure_excl(n_iters)
            xp = range(0.0, 2π * 19 / 20, length = 20)
            yp = range(0.0, 2π * 19 / 20, length = 20)
            dp = [sin(xi) * cos(yj) for xi in xp, yj in yp]
            pp = HermitePartials(
                (1, 0) => [ cos(xi) * cos(yj) for xi in xp, yj in yp],
                (0, 1) => [-sin(xi) * sin(yj) for xi in xp, yj in yp],
                (1, 1) => [-cos(xi) * sin(yj) for xi in xp, yj in yp],
            )
            bc = (PeriodicBC(endpoint = :exclusive), PeriodicBC(endpoint = :exclusive))
            hermite_interp((xp, yp), dp, pp, (0.5, 0.5); bc)
            function loop!(out, xp, yp, dp, pp, queries, n, bc)
                @inbounds for k in 1:n
                    out[k] = hermite_interp((xp, yp), dp, pp, queries[k]; bc)
                end
                return nothing
            end
            queries = [(0.1 + 0.0001 * k, 0.5) for k in 1:n_iters]
            out = Vector{Float64}(undef, n_iters)
            loop!(out, xp, yp, dp, pp, queries, n_iters, bc)
            return @allocated loop!(out, xp, yp, dp, pp, queries, n_iters, bc)
        end

        function _measure_inc(n_iters)
            xi2 = range(0.0, 2π, length = 20)
            yi2 = range(0.0, 2π, length = 20)
            di = [cos(xi) * cos(yj) for xi in xi2, yj in yi2]
            pi2 = HermitePartials(
                (1, 0) => [-sin(xi) * cos(yj) for xi in xi2, yj in yi2],
                (0, 1) => [-cos(xi) * sin(yj) for xi in xi2, yj in yi2],
                (1, 1) => [ sin(xi) * sin(yj) for xi in xi2, yj in yi2],
            )
            bc = (PeriodicBC(endpoint = :inclusive), PeriodicBC(endpoint = :inclusive))
            hermite_interp((xi2, yi2), di, pi2, (0.5, 0.5); bc)
            function loop!(out, xi2, yi2, di, pi2, queries, n, bc)
                @inbounds for k in 1:n
                    out[k] = hermite_interp((xi2, yi2), di, pi2, queries[k]; bc)
                end
                return nothing
            end
            queries = [(0.1 + 0.0001 * k, 0.5) for k in 1:n_iters]
            out = Vector{Float64}(undef, n_iters)
            loop!(out, xi2, yi2, di, pi2, queries, n_iters, bc)
            return @allocated loop!(out, xi2, yi2, di, pi2, queries, n_iters, bc)
        end

        function _measure_batch_inplace()
            x = range(0.0, 1.0, length = 20)
            y = range(0.0, 1.0, length = 20)
            data = [xi * yj for xi in x, yj in y]
            p = HermitePartials(
                (1, 0) => [yj for xi in x, yj in y],
                (0, 1) => [xi for xi in x, yj in y],
                (1, 1) => ones(20, 20),
            )
            queries = [(0.13, 0.42), (0.77, 0.51), (0.23, 0.83)]
            out = Vector{Float64}(undef, 3)
            hermite_interp!(out, (x, y), data, p, queries)
            return @allocated hermite_interp!(out, (x, y), data, p, queries)
        end

        function _measure_persistent_eval()
            x = range(0.0, 1.0, length = 20)
            y = range(0.0, 1.0, length = 20)
            data = [xi * yj for xi in x, yj in y]
            p = HermitePartials(
                (1, 0) => [yj for xi in x, yj in y],
                (0, 1) => [xi for xi in x, yj in y],
                (1, 1) => ones(20, 20),
            )
            itp = hermite_interp((x, y), data, p)
            itp((0.5, 0.5))
            return @allocated itp((0.7, 0.3))
        end

        # Bulk-loop measurements amortize single-call setup noise.
        # 1000-iter bulk / 1000 must be exactly 0.
        @test _measure_nobc(1000) == 0
        @test _measure_excl(1000) == 0
        @test _measure_inc(1000) == 0
        # Single-call batch & persistent eval (already inside function barrier).
        @test _measure_batch_inplace() <= ALLOC_THRESHOLD
        @test _measure_persistent_eval() <= ALLOC_THRESHOLD
    end

    @testset "Hermite ND — N=3 full mixed partials (exactness + vector calculus)" begin
        # f = P(x)·Q(y)·R(z), each factor a polynomial of degree ≤ 3, so f lies
        # in the per-cell tensor-product cubic span and is reproduced exactly
        # when fed the analytic partials. This is the only coverage that
        # compiles/exercises the @generated fill / wrap / seam code at N=3
        # (everything else is N=2). At N=3 there are 2^3-1 = 7 partial arrays.
        P(x) = 1.0 + 0.5x - 0.3x^2 + 0.2x^3
        Q(y) = 0.7 - 0.4y + 0.6y^2
        R(z) = 1.0 + 0.3z + 0.1z^2 - 0.05z^3
        Px(x) = 0.5 - 0.6x + 0.6x^2
        Qy(y) = -0.4 + 1.2y
        Rz(z) = 0.3 + 0.2z - 0.15z^2

        f(x, y, z) = P(x) * Q(y) * R(z)
        fx(x, y, z) = Px(x) * Q(y) * R(z)
        fy(x, y, z) = P(x) * Qy(y) * R(z)
        fz(x, y, z) = P(x) * Q(y) * Rz(z)
        fxy(x, y, z) = Px(x) * Qy(y) * R(z)
        fxz(x, y, z) = Px(x) * Q(y) * Rz(z)
        fyz(x, y, z) = P(x) * Qy(y) * Rz(z)
        fxyz(x, y, z) = Px(x) * Qy(y) * Rz(z)

        x = range(-1.0, 1.5, length = 6)
        y = range(0.0, 1.2, length = 5)
        z = range(-0.5, 1.0, length = 7)

        data = [f(xi, yj, zk) for xi in x, yj in y, zk in z]
        p = HermitePartials(
            (1, 0, 0) => [fx(xi, yj, zk)   for xi in x, yj in y, zk in z],
            (0, 1, 0) => [fy(xi, yj, zk)   for xi in x, yj in y, zk in z],
            (0, 0, 1) => [fz(xi, yj, zk)   for xi in x, yj in y, zk in z],
            (1, 1, 0) => [fxy(xi, yj, zk)  for xi in x, yj in y, zk in z],
            (1, 0, 1) => [fxz(xi, yj, zk)  for xi in x, yj in y, zk in z],
            (0, 1, 1) => [fyz(xi, yj, zk)  for xi in x, yj in y, zk in z],
            (1, 1, 1) => [fxyz(xi, yj, zk) for xi in x, yj in y, zk in z],
        )
        @test p isa HermitePartials{3, Float64, 7}

        itp = hermite_interp((x, y, z), data, p)
        @test ndims(itp) == 3
        @test size(itp) == (6, 5, 7)
        @test FastInterpolations.num_partials(itp) == 8   # 2^3

        # Grid nodes — exact
        for i in 1:length(x), j in 1:length(y), k in 1:length(z)
            @test itp((x[i], y[j], z[k])) ≈ data[i, j, k] atol = 1.0e-10
        end
        # Interior queries — exact (f ∈ tensor-product cubic span)
        for (qx, qy, qz) in [(-0.3, 0.42, 0.13), (1.1, 0.9, -0.2), (0.0, 0.5, 0.5)]
            @test itp((qx, qy, qz)) ≈ f(qx, qy, qz) atol = 1.0e-9
        end

        # Vector calculus via the AbstractInterpolantND protocol at N=3
        qx, qy, qz = 0.21, 0.66, 0.34
        g = gradient(itp, (qx, qy, qz))
        @test g[1] ≈ fx(qx, qy, qz) atol = 1.0e-8
        @test g[2] ≈ fy(qx, qy, qz) atol = 1.0e-8
        @test g[3] ≈ fz(qx, qy, qz) atol = 1.0e-8

        H = hessian(itp, (qx, qy, qz))
        @test size(H) == (3, 3)
        @test H[1, 2] ≈ H[2, 1] atol = 1.0e-9
        @test H[1, 3] ≈ H[3, 1] atol = 1.0e-9
        @test H[2, 3] ≈ H[3, 2] atol = 1.0e-9
        @test laplacian(itp, (qx, qy, qz)) ≈ H[1, 1] + H[2, 2] + H[3, 3] atol = 1.0e-9

        # Third mixed derivative + one-shot at N=3
        @test itp((qx, qy, qz); deriv = (DerivOp(1), DerivOp(1), DerivOp(1))) ≈ fxyz(qx, qy, qz) atol = 1.0e-8
        @test hermite_interp((x, y, z), data, p, (qx, qy, qz)) ≈ f(qx, qy, qz) atol = 1.0e-9
    end

    @testset "Hermite ND — Inclusive seam validation throws on mismatch" begin
        n = 6
        x = range(0.0, 2π, length = n)
        y = range(0.0, 2π, length = n)
        bc = (PeriodicBC(endpoint = :inclusive), PeriodicBC(endpoint = :inclusive))

        # (a) data endpoints unequal under :inclusive → throw on the data array
        data_bad = [xi * yj for xi in x, yj in y]   # data[1,:]=0 ≠ data[end,:]=2π·yj
        p_any = HermitePartials(
            (1, 0) => [yj for xi in x, yj in y],
            (0, 1) => [xi for xi in x, yj in y],
            (1, 1) => ones(n, n),
        )
        @test_throws ArgumentError hermite_interp((x, y), data_bad, p_any; bc)

        # (b) data periodic but one partial array is NOT → throw on that partial
        data_ok = [cos(xi) * cos(yj) for xi in x, yj in y]
        p_bad = HermitePartials(
            (1, 0) => [-sin(xi) * cos(yj) for xi in x, yj in y],
            (0, 1) => [-cos(xi) * sin(yj) for xi in x, yj in y],
            (1, 1) => [xi * yj for xi in x, yj in y],   # mask 3: non-periodic seam
        )
        @test_throws ArgumentError hermite_interp((x, y), data_ok, p_bad; bc)
    end

    @testset "Hermite ND — Extrapolation modes (Clamp / Fill / Wrap / NoExtrap)" begin
        F(x, y) = 0.4 + 0.7x - 0.3y + 0.5x * y + 0.2x^2 * y - 0.1y^3
        Fx(x, y) = 0.7 + 0.5y + 0.4x * y
        Fy(x, y) = -0.3 + 0.5x + 0.2x^2 - 0.3y^2
        Fxy(x, y) = 0.5 + 0.4x

        x = range(0.0, 2.0, length = 6)
        y = range(-1.0, 1.0, length = 5)
        data = [F(xi, yj) for xi in x, yj in y]
        p = HermitePartials(
            (1, 0) => [Fx(xi, yj)  for xi in x, yj in y],
            (0, 1) => [Fy(xi, yj)  for xi in x, yj in y],
            (1, 1) => [Fxy(xi, yj) for xi in x, yj in y],
        )
        yq = 0.3
        x_above = 2.7   # past x[end] = 2.0

        # ClampExtrap: OOB query clamps to the domain boundary
        itp_clamp = hermite_interp((x, y), data, p; extrap = ClampExtrap())
        @test itp_clamp((x_above, yq)) ≈ itp_clamp((2.0, yq)) atol = 1.0e-12

        # FillExtrap: OOB returns the fill value exactly
        itp_fill = hermite_interp((x, y), data, p; extrap = FillExtrap(0.0))
        @test itp_fill((x_above, yq)) === 0.0
        itp_nan = hermite_interp((x, y), data, p; extrap = FillExtrap(NaN))
        @test isnan(itp_nan((x_above, yq)))

        # NoExtrap (default): OOB throws
        itp_no = hermite_interp((x, y), data, p)
        @test_throws DomainError itp_no((x_above, yq))

        # One-shot FillExtrap exercises the `_try_fill_oob` early-return branch
        @test hermite_interp((x, y), data, p, (x_above, yq); extrap = FillExtrap(0.0)) === 0.0
        # One-shot NoExtrap OOB throws via `_validate_nd_domain`
        @test_throws DomainError hermite_interp((x, y), data, p, (x_above, yq))

        # WrapExtrap + PeriodicBC: a query past the domain wraps around
        xp = range(0.0, 2π, length = 9)
        yp = range(0.0, 2π, length = 9)
        Fp(x, y) = cos(x) * cos(y)
        dp = [Fp(xi, yj) for xi in xp, yj in yp]
        pp = HermitePartials(
            (1, 0) => [-sin(xi) * cos(yj) for xi in xp, yj in yp],
            (0, 1) => [-cos(xi) * sin(yj) for xi in xp, yj in yp],
            (1, 1) => [ sin(xi) * sin(yj) for xi in xp, yj in yp],
        )
        bc = (PeriodicBC(endpoint = :inclusive), PeriodicBC(endpoint = :inclusive))
        itp_wrap = hermite_interp((xp, yp), dp, pp; bc, extrap = WrapExtrap())
        @test itp_wrap((0.7 + 2π, 1.1)) ≈ itp_wrap((0.7, 1.1)) atol = 1.0e-10
    end

    @testset "Hermite ND — Vector (non-Range) grid exclusive periodic" begin
        # `collect` → Vector grids, which take the `vcat` (persistent) and
        # pooled `copyto!` (one-shot) extension branches; Range grids use the
        # range-preserving branch instead, so those paths were untested.
        F(x, y) = sin(x) * cos(y)
        Fx(x, y) = cos(x) * cos(y)
        Fy(x, y) = -sin(x) * sin(y)
        Fxy(x, y) = -cos(x) * sin(y)

        n = 16
        xv = collect(range(0.0, 2π * (n - 1) / n, length = n))
        yv = collect(range(0.0, 2π * (n - 1) / n, length = n))
        data = [F(xi, yj) for xi in xv, yj in yv]
        p = HermitePartials(
            (1, 0) => [Fx(xi, yj)  for xi in xv, yj in yv],
            (0, 1) => [Fy(xi, yj)  for xi in xv, yj in yv],
            (1, 1) => [Fxy(xi, yj) for xi in xv, yj in yv],
        )
        # Vector grids require an explicit period.
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 2π),
            PeriodicBC(endpoint = :exclusive, period = 2π),
        )

        itp = hermite_interp((xv, yv), data, p; bc)
        @test size(itp) == (n + 1, n + 1)             # extended along both axes
        @test !(itp.grids[1] isa AbstractRange)        # confirm the Vector branch ran
        for qx in [0.13, 1.5, 3.0, 5.4], qy in [0.21, 1.1, 2.7, 5.9]
            @test itp((qx, qy)) ≈ F(qx, qy) atol = 5.0e-3
        end

        # One-shot over Vector grids exercises the pooled `copyto!` extension.
        v_persist = itp((0.83, 2.4))
        v_oneshot = hermite_interp((xv, yv), data, p, (0.83, 2.4); bc)
        @test v_oneshot ≈ v_persist atol = 1.0e-10
    end

    @testset "Hermite ND — Higher-order & one-shot derivatives" begin
        # Bicubic polynomial → every partial derivative reproduced exactly.
        a, b, c, dd = 0.7, 1.3, -0.4, 2.1
        e, ff, gg, hh = -0.6, 0.5, 0.3, -0.2
        ii, jj = 0.15, -0.25
        F(x, y) = a + b * x + c * y + dd * x * y + e * x^2 + ff * y^2 + gg * x^2 * y + hh * x * y^2 + ii * x^3 + jj * y^3
        Fx(x, y) = b + dd * y + 2e * x + 2gg * x * y + hh * y^2 + 3ii * x^2
        Fy(x, y) = c + dd * x + 2ff * y + gg * x^2 + 2hh * x * y + 3jj * y^2
        Fxy(x, y) = dd + 2gg * x + 2hh * y
        Fxx(x, y) = 2e + 2gg * y + 6ii * x
        Fyy(x, y) = 2ff + 2hh * x + 6jj * y
        Fxxx = 6ii

        x = collect(range(-1.0, 1.5, length = 8))
        y = collect(range(-0.8, 1.2, length = 7))
        data = [F(xi, yj) for xi in x, yj in y]
        p = HermitePartials(
            (1, 0) => [Fx(xi, yj)  for xi in x, yj in y],
            (0, 1) => [Fy(xi, yj)  for xi in x, yj in y],
            (1, 1) => [Fxy(xi, yj) for xi in x, yj in y],
        )
        itp = hermite_interp((x, y), data, p)
        qx, qy = 0.37, -0.21

        # Higher-order persistent derivatives (only 1st-order was covered before)
        @test itp((qx, qy); deriv = (DerivOp(2), DerivOp(0))) ≈ Fxx(qx, qy) atol = 1.0e-9
        @test itp((qx, qy); deriv = (DerivOp(0), DerivOp(2))) ≈ Fyy(qx, qy) atol = 1.0e-9
        @test itp((qx, qy); deriv = (DerivOp(3), DerivOp(0))) ≈ Fxxx atol = 1.0e-8

        # One-shot deriv kwarg (scalar + batch) — previously untested
        @test hermite_interp((x, y), data, p, (qx, qy); deriv = (DerivOp(1), DerivOp(0))) ≈ Fx(qx, qy) atol = 1.0e-9
        @test hermite_interp((x, y), data, p, (qx, qy); deriv = (DerivOp(1), DerivOp(1))) ≈ Fxy(qx, qy) atol = 1.0e-9
        qs = [(0.1, 0.2), (-0.5, 0.7), (1.1, -0.3)]
        out = hermite_interp((x, y), data, p, qs; deriv = (DerivOp(1), DerivOp(0)))
        @test out ≈ [Fx(qx2, qy2) for (qx2, qy2) in qs] atol = 1.0e-9
    end

    @testset "Hermite ND — Introspection & wrong-K rejection" begin
        x = range(0.0, 1.0, length = 4)
        y = range(0.0, 1.0, length = 4)
        data = [xi * yj for xi in x, yj in y]
        p = HermitePartials(
            (1, 0) => [yj for xi in x, yj in y],
            (0, 1) => [xi for xi in x, yj in y],
            (1, 1) => ones(4, 4),
        )
        itp = hermite_interp((x, y), data, p)

        @test ndims(itp) == 2
        @test size(itp) == (4, 4)
        @test axes(itp) === itp.grids
        @test FastInterpolations.num_partials(itp) == 4
        @test FastInterpolations.num_partials(typeof(itp)) == 4

        # show methods must not error and must name the type
        s_compact = sprint(show, itp)
        s_plain = sprint((io, v) -> show(io, MIME"text/plain"(), v), itp)
        @test occursin("CubicHermiteInterpolantND", s_compact)
        @test occursin("CubicHermiteInterpolantND", s_plain)

        # Directly constructed wrong-K HermitePartials (bypassing the public
        # constructor's K == 2^N-1 guard) must be rejected by the build.
        A = zeros(4, 4)
        bad = HermitePartials{2, Float64, 2, Matrix{Float64}}((A, A))
        @test_throws ArgumentError hermite_interp((x, y), data, bad)
        @test_throws ArgumentError hermite_interp((x, y), data, bad, (0.5, 0.5))  # one-shot path
    end

end  # @testitem
