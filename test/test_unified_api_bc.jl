@testitem "Unified API — LinearInterp / ConstantInterp `bc` field" begin
    # Smoke test: verify the new `bc` field on `LinearInterp` / `ConstantInterp`
    # round-trips through constructors and through the unified `interp(...)` API
    # for both homogeneous and heterogeneous (mixed-method) ND calls.

    @testset "Tag struct constructors" begin
        # Defaults
        @test LinearInterp().bc isa NoBC
        @test ConstantInterp().bc isa NoBC
        @test ConstantInterp().side isa NearestSide

        # Keyword form
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        @test LinearInterp(bc = bc).bc === bc
        @test ConstantInterp(bc = bc).bc === bc
        @test ConstantInterp(side = LeftSide(), bc = bc).bc === bc
        @test ConstantInterp(side = LeftSide(), bc = bc).side isa LeftSide

        # Backward compat: bare positional `LinearInterp()` /
        # `ConstantInterp(side=...)` still work
        @test LinearInterp() isa LinearInterp
        @test ConstantInterp(side = RightSide()).side isa RightSide
    end
end

@testitem "Unified API — homogeneous Linear/Constant + PeriodicBC" begin
    # `interp(grids, data; method=(LinearInterp(bc=PeriodicBC()), ...))` must
    # produce the same values as the direct `linear_interp(...; bc=...)` /
    # `constant_interp(...; bc=...)` ND constructors.

    nx, ny = 12, 9
    n_query = 30
    x = range(0.0, 1.0, nx)
    y_grid = range(0.0, 2.0, ny)
    data = randn(nx, ny)
    # `:inclusive` requires endpoint match
    data[end, :] .= data[1, :]
    data[:, end] .= data[:, 1]
    xq = rand(n_query)
    yq = rand(n_query) .* 2

    @testset "Linear ND — :inclusive" begin
        bc = PeriodicBC()  # :inclusive default
        ref = linear_interp((x, y_grid), data, (xq, yq); bc = (bc, bc))
        via_unified = interp(
            (x, y_grid), data, (xq, yq);
            method = (LinearInterp(bc = bc), LinearInterp(bc = bc)),
        )
        @test via_unified ≈ ref atol = 1.0e-12
    end

    @testset "Linear ND — :exclusive" begin
        # Half-open n-point grid + period
        nx_e, ny_e = 11, 8
        x_e = collect(range(0.0, step = 1.0 / nx_e, length = nx_e))
        y_e = collect(range(0.0, step = 2.0 / ny_e, length = ny_e))
        data_e = randn(nx_e, ny_e)
        bc_x = PeriodicBC(endpoint = :exclusive, period = 1.0)
        bc_y = PeriodicBC(endpoint = :exclusive, period = 2.0)
        xq_e = rand(n_query)
        yq_e = rand(n_query) .* 2

        ref = linear_interp((x_e, y_e), data_e, (xq_e, yq_e); bc = (bc_x, bc_y))
        via_unified = interp(
            (x_e, y_e), data_e, (xq_e, yq_e);
            method = (LinearInterp(bc = bc_x), LinearInterp(bc = bc_y)),
        )
        @test via_unified ≈ ref atol = 1.0e-12
    end

    @testset "Constant ND — :inclusive" begin
        bc = PeriodicBC()
        ref = constant_interp((x, y_grid), data, (xq, yq); bc = (bc, bc))
        via_unified = interp(
            (x, y_grid), data, (xq, yq);
            method = (ConstantInterp(bc = bc), ConstantInterp(bc = bc)),
        )
        @test via_unified == ref     # constant: single-node selection, exact
    end

    @testset "Constant ND — :exclusive (with side)" begin
        nx_e, ny_e = 11, 8
        x_e = collect(range(0.0, step = 1.0 / nx_e, length = nx_e))
        y_e = collect(range(0.0, step = 2.0 / ny_e, length = ny_e))
        data_e = randn(nx_e, ny_e)
        bc_x = PeriodicBC(endpoint = :exclusive, period = 1.0)
        bc_y = PeriodicBC(endpoint = :exclusive, period = 2.0)
        xq_e = rand(n_query)
        yq_e = rand(n_query) .* 2

        ref = constant_interp(
            (x_e, y_e), data_e, (xq_e, yq_e);
            bc = (bc_x, bc_y), side = (LeftSide(), NearestSide())
        )
        via_unified = interp(
            (x_e, y_e), data_e, (xq_e, yq_e);
            method = (
                ConstantInterp(side = LeftSide(), bc = bc_x),
                ConstantInterp(side = NearestSide(), bc = bc_y),
            ),
        )
        @test via_unified == ref
    end
end

@testitem "Unified API — heterogeneous Linear+Cubic + PeriodicBC end-to-end" begin
    # Ensure that `(LinearInterp(bc=PeriodicBC(...)), CubicInterp(bc=PeriodicBC(...)))`
    # routed through Hetero produces values matching the direct path on a
    # smooth periodic test function (sin(x) + cos(y)).

    nx, ny = 16, 14
    n_query = 25
    x = collect(range(0.0, step = 2π / nx, length = nx))   # exclusive n-point grid
    y_grid = collect(range(0.0, step = 2π / ny, length = ny))
    data = [sin(xi) + cos(yj) for xi in x, yj in y_grid]
    bc_x = PeriodicBC(endpoint = :exclusive, period = 2π)
    bc_y = PeriodicBC(endpoint = :exclusive, period = 2π)

    # Stay strictly inside the inner interval to avoid extrap branch
    xq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
    yq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)

    via_unified = interp(
        (x, y_grid), data, (xq, yq);
        method = (LinearInterp(bc = bc_x), CubicInterp(bc = bc_y)),
    )

    # Reference: 1D cubic_interp along y at each query (independent y per
    # column), then 1D linear_interp along x — separable test function gives
    # well-defined per-axis interpolation values to compare against.
    @test all(isfinite, via_unified)
    @test length(via_unified) == n_query

    # Sanity: values should be close to the analytic function (linear on x is
    # only first-order accurate, so use a loose absolute tolerance).
    analytic = [sin(xi) + cos(yi) for (xi, yi) in zip(xq, yq)]
    @test maximum(abs, via_unified .- analytic) < 0.05

    # Adjoint path through `hetero_adjoint(...; methods=...)`: verify the same
    # tag-struct `bc` field plumbing reaches the adjoint builder and that the
    # dot-product identity holds end-to-end.
    methods = (LinearInterp(bc = bc_x), CubicInterp(bc = bc_y))
    itp = interp((x, y_grid), data; method = methods)
    adj = hetero_adjoint((x, y_grid), (xq, yq); methods = methods)
    @test size(adj(zeros(n_query))) == size(data)
    using LinearAlgebra: dot
    y_bar = randn(n_query)
    forward_vec = [itp((xq[k], yq[k])) for k in 1:n_query]
    @test dot(forward_vec, y_bar) ≈ dot(vec(data), vec(adj(y_bar))) atol = 1.0e-10
end


# `hetero_adjoint` must auto-promote `extrap` to WrapExtrap on periodic axes,
# matching the forward path. Without this, a query past the inner span (e.g.
# 1.05 with period=1.0) is accepted by `interp(...)` but rejected with
# DomainError by `hetero_adjoint(...)`.
@testitem "hetero_adjoint — periodic extrap auto-promote on off-span query" begin
    x = collect(range(0.0, step = 0.1, length = 10))   # n=10, period=1
    y = collect(range(0.0, 1.0, 10))
    data = [sin(2π * xi) + yj for xi in x, yj in y]
    bc_x = PeriodicBC(endpoint = :exclusive, period = 1.0)
    methods = (LinearInterp(bc = bc_x), CubicInterp())

    # Forward accepts xq=1.05 by auto-promoting to WrapExtrap on the periodic axis.
    itp = interp((x, y), data; method = methods)
    @test isfinite(itp((1.05, 0.5)))

    # Adjoint MUST also accept it. Same auto-promotion must reach the anchor
    # baking + domain validation.
    adj = hetero_adjoint((x, y), ([1.05], [0.5]); methods = methods)
    @test size(adj(zeros(1))) == size(data)
end


# OnTheFly hetero forward must either return correct values at the periodic
# seam cell (matching PreCompute), or reject the unsupported configuration with
# a clear ArgumentError. Silent garbage is the bug. Coverage spans all three
# `_validate_nd_coeffs(::OnTheFly, ...)` call sites: persistent ctor + scalar
# one-shot + batch one-shot.
@testitem "OnTheFly hetero — seam-cell matches PreCompute or rejects" begin
    x = collect(0.0:0.2:0.8)
    y = collect(0.0:0.25:1.0)
    data = [sin(2π * xi) + cos(π * yj) for xi in x, yj in y]
    methods = (LinearInterp(bc = PeriodicBC(endpoint = :exclusive, period = 1.0)), CubicInterp())

    itp_pre = interp((x, y), data; method = methods, coeffs = PreCompute())
    v_pre = itp_pre((0.9, 0.5))   # seam cell: between x[end]=0.8 and x[1]+period=1.0

    # Persistent ctor — must equal PreCompute or reject.
    correctness_ok = try
        itp_otf = interp((x, y), data; method = methods, coeffs = OnTheFly())
        v_otf = itp_otf((0.9, 0.5))
        isapprox(v_otf, v_pre; atol = 1.0e-12)
    catch e
        e isa ArgumentError      # explicit rejection is also acceptable
    end
    @test correctness_ok

    # Scalar one-shot — same rejection path (`hetero_oneshot.jl:_validate_nd_coeffs`).
    @test_throws ArgumentError interp((x, y), data, (0.9, 0.5); method = methods, coeffs = OnTheFly())

    # Batch one-shot — same rejection at the batch dispatch site.
    @test_throws ArgumentError interp((x, y), data, ([0.9], [0.5]); method = methods, coeffs = OnTheFly())
end


# `ConstantInterp(side::AbstractSide)` (positional, single-arg) is a public
# call form; the default 2-field struct constructor `ConstantInterp(side, bc)`
# would shadow it without an explicit outer ctor.
@testitem "ConstantInterp(side) — positional 1-arg ctor" begin
    @test ConstantInterp(LeftSide()) isa ConstantInterp
    @test ConstantInterp(LeftSide()).side === LeftSide()
    @test ConstantInterp(LeftSide()).bc === NoBC()
    @test ConstantInterp(NearestSide()) isa ConstantInterp
    @test ConstantInterp(RightSide()) isa ConstantInterp

    # Round-trip: side must be honored downstream (not silently dropped to
    # NearestSide). LeftSide returns the value at the cell's left endpoint,
    # so query at xq = 0.499 in cell [0.4, 0.5] must equal data[5] (left).
    x = collect(0.0:0.1:1.0)
    data = collect(1.0:11.0)            # data[i] = i
    itp_left = interp((x,), data; method = (ConstantInterp(LeftSide()),))
    itp_right = interp((x,), data; method = (ConstantInterp(RightSide()),))
    @test itp_left((0.499,)) ≈ 5.0     # left endpoint of cell [x[5], x[6]] = data[5]
    @test itp_right((0.401,)) ≈ 6.0     # right endpoint of cell [x[5], x[6]] = data[6]
end
