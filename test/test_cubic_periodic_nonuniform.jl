# Regression: PeriodicBC cubic spline on NON-uniform grids must solve the true
# cyclic system h_{i-1}z_{i-1} + 2(h_{i-1}+h_i)z_i + h_i z_{i+1} = rhs_i.
# (The super-diagonal was once shifted by one cell: invisible on uniform grids,
# O(h³) values / O(h²) slopes on non-uniform ones.)

@testsnippet DensePeriodicSpline begin
    using LinearAlgebra

    # Closed-cycle non-uniform grid on [0, 2π] (x[end] == 2π)
    nonuniform_periodic_grid(n) = [2π * (t + 0.15 * sin(2π * t)) for t in range(0.0, 1.0, n)]

    # Dense reference: full m×m cyclic solve for the moments, returns q -> (S, S', S'')
    function dense_periodic_spline(x, y)
        m = length(x) - 1
        h = diff(x)
        A = zeros(m, m)
        r = zeros(m)
        for i in 1:m
            im = i == 1 ? m : i - 1
            ip = i == m ? 1 : i + 1
            A[i, im] += h[im]
            A[i, i] += 2 * (h[im] + h[i])
            A[i, ip] += h[i]
            ym = i == 1 ? y[m] : y[i - 1]
            r[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - ym) / h[im])
        end
        M = A \ r
        push!(M, M[1])
        return function (q)
            k = clamp(searchsortedlast(x, q), 1, m)
            hk = h[k]
            a = x[k + 1] - q
            b = q - x[k]
            c0 = y[k] / hk - M[k] * hk / 6
            c1 = y[k + 1] / hk - M[k + 1] * hk / 6
            v = (M[k] * a^3 + M[k + 1] * b^3) / (6hk) + c0 * a + c1 * b
            d1 = (-M[k] * a^2 + M[k + 1] * b^2) / (2hk) - c0 + c1
            d2 = (M[k] * a + M[k + 1] * b) / hk
            return (v, d1, d2)
        end
    end
end

@testitem "Cubic PeriodicBC non-uniform: matches dense cyclic solve" setup = [DensePeriodicSpline] begin
    x = nonuniform_periodic_grid(41)
    y = sin.(x)
    y[end] = y[1]   # exact closed cycle (Series checks it strictly)
    ref = dense_periodic_spline(x, y)
    xq = range(0.0, 2π, 397)
    rv = [ref(q) for q in xq]

    for itp in (
            cubic_interp(x, y; bc = PeriodicBC()),
            cubic_interp(x[1:(end - 1)], y[1:(end - 1)]; bc = PeriodicBC(endpoint = :exclusive, period = 2π)),
        )
        for (k, op) in enumerate((EvalValue(), DerivOp(1), DerivOp(2)))
            @test maximum(abs(itp(q; deriv = op) - rv[j][k]) for (j, q) in enumerate(xq)) < 1.0e-12
        end
    end

    # One-shot and Series paths share the cache
    @test maximum(abs.(cubic_interp(x, y, collect(xq); bc = PeriodicBC()) .- first.(rv))) < 1.0e-12
    sitp = cubic_interp(x, Series(y, 2 .* y); bc = PeriodicBC())
    @test maximum(maximum(abs.(sitp(q) .- (1, 2) .* rv[j][1])) for (j, q) in enumerate(xq)) < 1.0e-12

    # ND: periodic non-uniform axis × linear data on a CubicFit axis (reproduced exactly)
    x2 = collect(range(0.0, 1.0, 6))
    data = [sin(xi) * (1 + 2 * yj) for xi in x, yj in x2]
    itp2 = cubic_interp((x, x2), data; bc = (PeriodicBC(), CubicFit()))
    @test maximum(abs(itp2((q, 0.3)) - rv[j][1] * 1.6) for (j, q) in enumerate(xq)) < 1.0e-12
end

@testitem "Cubic PeriodicBC non-uniform: convergence orders" setup = [DensePeriodicSpline] begin
    function interior_errors(n)
        x = nonuniform_periodic_grid(n)
        itp = cubic_interp(x, sin.(x); bc = PeriodicBC())
        xq = range(0.0, 2π, 2001)
        ev = maximum(abs(itp(q) - sin(q)) for q in xq)
        es = maximum(abs(itp(q; deriv = DerivOp(1)) - cos(q)) for q in xq)
        return ev, es
    end
    (ev1, es1), (ev2, es2) = interior_errors(65), interior_errors(129)
    @test log2(ev1 / ev2) > 3.7   # O(h⁴) values
    @test log2(es1 / es2) > 2.7   # O(h³) slopes
end

@testitem "Cubic PeriodicBC non-uniform: adjoint matches dense Wᵀ" setup = [DensePeriodicSpline] begin
    x = nonuniform_periodic_grid(21)
    m = length(x) - 1
    xq = collect(range(0.05, 2π - 0.05, 17))
    ȳ = cos.(3 .* xq) .+ 0.5

    # Dense W on the m periodic DOFs (column j = spline of periodic unit vector e_j)
    W = zeros(length(xq), m)
    for j in 1:m
        e = zeros(m + 1)
        e[j] = 1
        j == 1 && (e[end] = 1)
        ref = dense_periodic_spline(x, e)
        W[:, j] = [ref(q)[1] for q in xq]
    end

    f̄ = cubic_adjoint(x, xq; bc = PeriodicBC())(ȳ)
    f̄_periodic = vcat(f̄[1] + f̄[end], f̄[2:(end - 1)])   # fold the duplicated seam node
    @test f̄_periodic ≈ W' * ȳ rtol = 1.0e-12
end
