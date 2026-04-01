# ALLOC_THRESHOLD is defined in runtests.jl

@testset "PCHIP 1D" begin

    # ========================================
    # Test data
    # ========================================
    # Cubic polynomial: PCHIP should be close (not exact — weighted harmonic mean
    # introduces small errors vs exact cubic polynomial derivatives)
    p(x) = 2x^3 - x^2 + 3x - 1
    dp(x) = 6x^2 - 2x + 3

    @testset "Basic correctness — oneshot scalar" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            val = pchip_interp(x, y, xq)
            @test val ≈ p(xq) atol = 1.0e-2
        end
    end

    @testset "Basic correctness — oneshot vector" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        result = pchip_interp(x, y, xq)
        @test result ≈ p.(xq) atol = 1.0e-2
    end

    @testset "Basic correctness — in-place" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        out = zeros(length(xq))
        pchip_interp!(out, x, y, xq)
        @test out ≈ p.(xq) atol = 1.0e-2
    end

    @testset "Basic correctness — callable interpolant" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)

        itp = pchip_interp(x, y)
        @test itp isa PchipInterpolant1D

        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            @test itp(xq) ≈ p(xq) atol = 1.0e-2
        end

        # Vector call
        @test itp(xq_pts) ≈ p.(xq_pts) atol = 1.0e-2

        # In-place call
        out = zeros(length(xq_pts))
        itp(out, xq_pts)
        @test out ≈ p.(xq_pts) atol = 1.0e-2
    end

    @testset "Monotonicity guarantee" begin
        # Strictly monotone increasing input
        x = collect(range(0.0, 5.0, 10))
        y = cumsum(rand(MersenneTwister(42), 10))  # guaranteed monotone increasing

        itp = pchip_interp(x, y)

        # Dense sampling — interpolant must be monotone
        xq_dense = collect(range(first(x), last(x), 1000))
        yq_dense = itp(xq_dense)
        diffs = diff(yq_dense)
        @test all(d -> d >= -eps(), diffs)  # allow tiny numerical noise

        # Strictly monotone decreasing input
        y_dec = reverse(y)
        itp_dec = pchip_interp(x, y_dec)
        yq_dec = itp_dec(xq_dense)
        diffs_dec = diff(yq_dec)
        @test all(d -> d <= eps(), diffs_dec)
    end

    @testset "Local extrema — slope clamping" begin
        # Data with clear local extremum at x=2
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 0.5, 1.0, 2.0]  # local min at x=2

        itp = pchip_interp(x, y)

        # Slopes at local extrema should be zero → derivative at x=2 should be ~0
        # (PCHIP zeros the slope when adjacent secants have opposite signs)
        d_at_min = itp(2.0; deriv = DerivOp(1))
        @test abs(d_at_min) < 1.0e-10
    end

    @testset "DerivOp correctness" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = 1.5

        # Value
        @test pchip_interp(x, y, xq; deriv = EvalValue()) ≈ p(xq) atol = 1.0e-2

        # First derivative (PCHIP slopes ≈ true derivatives, harmonic mean introduces error)
        @test pchip_interp(x, y, xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 0.05

        # Fourth+ derivative → zero
        @test pchip_interp(x, y, xq; deriv = DerivOp(4)) ≈ 0.0 atol = 1.0e-14

        # Via callable
        itp = pchip_interp(x, y)
        @test itp(xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 0.05
    end

    @testset "Extrapolation modes" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)

        # NoExtrap — throws outside domain
        @test_throws DomainError pchip_interp(x, y, -0.1)
        @test_throws DomainError pchip_interp(x, y, 3.1)

        # ClampExtrap — returns boundary values
        @test pchip_interp(x, y, -0.5; extrap = ClampExtrap()) ≈ y[1]
        @test pchip_interp(x, y, 3.5; extrap = ClampExtrap()) ≈ y[end]

        # ExtendExtrap — extends boundary polynomial
        val_ext = pchip_interp(x, y, 3.2; extrap = ExtendExtrap())
        @test isfinite(val_ext)
        @test val_ext ≈ p(3.2) atol = 0.5  # large tol: boundary slope ≠ exact derivative → extrapolation diverges

        # FillExtrap — returns fill value outside domain
        @test pchip_interp(x, y, -0.5; extrap = FillExtrap(999.0)) ≈ 999.0
        @test pchip_interp(x, y, 3.5; extrap = FillExtrap(NaN)) |> isnan
        # FillExtrap derivative at OOB → zero
        @test pchip_interp(x, y, -0.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1e-14
        @test pchip_interp(x, y, 3.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1e-14

        # WrapExtrap — wraps to domain
        val_wrap = pchip_interp(x, y, 3.2; extrap = WrapExtrap())
        @test isfinite(val_wrap)
        # WrapExtrap — vector fast-path includes x_max (no unnecessary wrap overhead)
        xq_boundary = [x[end]]
        out_boundary = similar(xq_boundary)
        pchip_interp!(out_boundary, x, y, xq_boundary; extrap = WrapExtrap())
        @test isfinite(out_boundary[1])

        # Callable with extrap
        itp = pchip_interp(x, y; extrap = ClampExtrap())
        @test itp(-0.5) ≈ y[1]
        @test itp(3.5) ≈ y[end]
    end

    @testset "Type stability — @inferred" begin
        x = collect(range(0.0, 3.0, 10))
        y = sin.(x)
        xq_scalar = 1.5
        xq_vec = [0.5, 1.5, 2.5]

        @test @inferred(pchip_interp(x, y, xq_scalar)) isa Float64
        @test @inferred(pchip_interp(x, y, xq_vec)) isa Vector{Float64}

        itp = pchip_interp(x, y)
        @test @inferred(itp(xq_scalar)) isa Float64
        @test @inferred(itp(xq_vec)) isa Vector{Float64}
    end

    @testset "Zero allocation" begin
        x = collect(range(0.0, 3.0, 50))
        y = sin.(x)

        # Scalar oneshot (uses @with_pool for dy buffer)
        function test_pchip_scalar_alloc(x, y, xq)
            pchip_interp(x, y, xq)
            return @allocated pchip_interp(x, y, xq)
        end
        @test test_pchip_scalar_alloc(x, y, 1.5) <= ALLOC_THRESHOLD

        # Callable scalar (slopes precomputed — pure eval)
        itp = pchip_interp(x, y)
        function test_pchip_callable_alloc(itp, xq)
            itp(xq)
            return @allocated itp(xq)
        end
        @test test_pchip_callable_alloc(itp, 1.5) == 0
    end

    @testset "Uniform grid (Range input)" begin
        x = range(0.0, 3.0, 20)
        y = p.(collect(x))

        # Callable with range grid — verify _CachedRange preserved
        itp = pchip_interp(collect(x), y)  # Note: pchip_interp promotes via _promote_itp_inputs
        @test itp(1.5) ≈ p(1.5) atol = 1.0e-2
    end

    @testset "Non-uniform grid" begin
        x = sort([0.0, 0.1, 0.3, 0.7, 1.2, 2.0, 2.5, 3.0])
        y = p.(x)

        @test pchip_interp(x, y, 1.5) ≈ p(1.5) atol = 0.5  # coarse non-uniform grid → larger error
    end

    @testset "2-point grid (minimum — linear)" begin
        x = [0.0, 1.0]
        y = [1.0, 3.0]

        # 2-point PCHIP → linear interpolation (both slopes = secant)
        @test pchip_interp(x, y, 0.5) ≈ 2.0 atol = 1.0e-12
    end

    @testset "3-point grid" begin
        x = [0.0, 1.0, 3.0]
        y = [1.0, 2.0, 0.5]

        val = pchip_interp(x, y, 1.5)
        @test isfinite(val)
    end

    @testset "Flat data — zero slopes" begin
        x = collect(range(0.0, 5.0, 10))
        y = fill(3.0, 10)

        itp = pchip_interp(x, y)
        @test itp(2.5) ≈ 3.0 atol = 1.0e-14
        @test itp(2.5; deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14
    end

    @testset "Integer input promotion" begin
        x_int = 0:5
        y_int = [0, 1, 4, 9, 16, 25]

        val = pchip_interp(collect(x_int), y_int, 2.5)
        @test val isa Float64
        @test isfinite(val)
    end

    @testset "Float32 precision" begin
        x32 = collect(range(0.0f0, 3.0f0, 20))
        y32 = sin.(x32)

        val = pchip_interp(x32, y32, 1.5f0)
        @test val isa Float32
        @test val ≈ Float32(sin(1.5)) atol = 1.0e-2

        itp32 = pchip_interp(x32, y32)
        @test itp32(1.5f0) isa Float32
    end

    @testset "Broadcast" begin
        x = collect(range(0.0, 3.0, 20))
        y = sin.(x)

        itp = pchip_interp(x, y)
        xq = [0.5, 1.0, 2.0]
        @test all(isfinite, itp.(xq))
    end

    @testset "Search policy override" begin
        x = collect(range(0.0, 3.0, 50))
        y = p.(x)
        xq = 1.5

        @test pchip_interp(x, y, xq; search = BinarySearch()) ≈ p(xq) atol = 1.0e-2
        @test pchip_interp(x, y, xq; search = LinearBinarySearch()) ≈ p(xq) atol = 1.0e-2

        itp = pchip_interp(x, y)
        @test itp(xq; search = BinarySearch()) ≈ p(xq) atol = 1.0e-2
    end

    @testset "PCHIP vs cubic spline" begin
        # On smooth data, PCHIP and spline should be similar
        x = collect(range(0.0, 2π, 20))
        y = sin.(x)
        xq = [0.5, 1.5, 2.5, 3.5, 4.5]

        pchip_vals = pchip_interp(x, y, xq)
        spline_vals = cubic_interp(x, y, xq)

        # Both should be close to sin (not identical — different slope methods)
        @test pchip_vals ≈ sin.(xq) atol = 1.0e-2
        @test spline_vals ≈ sin.(xq) atol = 1.0e-2
    end

    @testset "Show methods" begin
        x = collect(range(0.0, 3.0, 10))
        y = sin.(x)
        itp = pchip_interp(x, y)

        # Compact
        compact = sprint(show, itp)
        @test occursin("PchipInterpolant1D", compact)
        @test occursin("monotone", compact)

        # Verbose
        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("PchipInterpolant1D", verbose)
        @test occursin("PCHIP", verbose)
    end

    @testset "In-place hint forwarding and zero-alloc" begin
        x = collect(range(0.0, 4.0, 50))
        y = sin.(x)
        xq = collect(range(0.1, 3.9, 20))
        output = Vector{Float64}(undef, length(xq))

        # hint gets updated through in-place path
        hint = Ref(1)
        pchip_interp!(output, x, y, xq; hint)
        @test hint[] > 1

        # Verify correctness with hint
        ref = pchip_interp(x, y, xq)
        @test output ≈ ref

        # Zero-allocation in-place (function barrier)
        function test_pchip_inplace_alloc(output, x, y, xq)
            hint_local = Ref(1)
            pchip_interp!(output, x, y, xq; hint = hint_local)
            return @allocated pchip_interp!(output, x, y, xq; hint = hint_local)
        end
        @test test_pchip_inplace_alloc(output, x, y, xq) <= ALLOC_THRESHOLD
    end
end
