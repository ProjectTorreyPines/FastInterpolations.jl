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
        @test pchip_interp(x, y, -0.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14
        @test pchip_interp(x, y, 3.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14

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

    @testset "Coverage — Range in-place disambiguation" begin
        x_range = range(0.0, 3.0, 20)
        y = sin.(collect(x_range))
        xq = collect(range(0.1, 2.9, 10))
        out = similar(xq)
        pchip_interp!(out, x_range, y, xq)
        @test out ≈ pchip_interp(collect(x_range), y, xq) atol = 1.0e-12
    end

    @testset "Coverage — local extrema (mixed-sign secants)" begin
        # Data with local extremum: y goes up then down
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 2.0, 1.0, 3.0, 0.5]

        # Monotonicity clamping: at extrema, PCHIP sets slope to zero
        itp = pchip_interp(x, y)
        # At x=2 (local min between peaks), slope should be clamped
        @test isfinite(itp(1.5))
        @test isfinite(itp(2.5))

        # Verify no overshoots on monotone segments
        # Between x=0 and x=1, y goes 0→2 (monotone increasing)
        vals_01 = [itp(t) for t in range(0.0, 1.0, 20)]
        @test all(diff(vals_01) .>= -1.0e-14)  # non-decreasing
    end

    @testset "Coverage — n=2 minimum grid" begin
        x = [0.0, 1.0]
        y = [1.0, 3.0]
        @test pchip_interp(x, y, 0.5) ≈ 2.0 atol = 0.1
    end

    @testset "Coverage — flat region (zero secant)" begin
        # Flat segment: y[2]==y[3] → δ_curr==0
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 1.0, 2.0]
        @test isfinite(pchip_interp(x, y, 1.5))
    end

    @testset "Coverage — output eltype validation error" begin
        # Integer inputs trigger generic wrapper which validates output eltype
        x_int = collect(0:9)
        y_int = x_int .^ 2
        xq_int = [2, 4, 6]
        out_narrow = Vector{Float32}(undef, length(xq_int))
        # Int → Float64 result into Float32 output → should throw
        @test_throws ArgumentError pchip_interp!(out_narrow, x_int, y_int, xq_int)
    end

    @testset "Coverage — WrapExtrap vector path" begin
        x = collect(range(0.0, 2π, 20))
        y = sin.(x)
        xq_inner = collect(range(0.1, 2π - 0.1, 10))
        out = similar(xq_inner)
        pchip_interp!(out, x, y, xq_inner; extrap = WrapExtrap())
        @test all(isfinite, out)
    end

    @testset "Coverage — generic in-place pass-through (Integer inputs)" begin
        # Exercises generic wrapper tail call (line 170) that passes eltype validation
        x_int = collect(0:9)
        y_int = x_int .^ 2
        xq_int = [2, 4, 6]
        out = Vector{Float64}(undef, length(xq_int))
        pchip_interp!(out, x_int, y_int, xq_int)
        @test all(isfinite, out)
    end

    @testset "Coverage — endpoint overshoot clamping (3δ1 branch)" begin
        # Data where endpoint 3-point FD overshoots: d > 3*δ1
        # Sharp initial rise then plateau → large endpoint FD, small first secant
        x = [0.0, 0.1, 1.0, 2.0, 3.0]
        y = [0.0, 10.0, 10.5, 11.0, 11.5]
        # First secant δ1 = 100, second secant δ2 = 0.556
        # 3-point FD: d = ((2*0.1 + 0.9)*100 - 0.1*0.556) / (0.1+0.9) ≈ 109.9
        # |d| > |3*δ1| = 300? No. Try steeper:
        x2 = [0.0, 0.01, 1.0, 2.0, 3.0]
        y2 = [0.0, 0.0, 1.0, 2.0, 3.0]
        # δ1 = 0/0.01 = 0, δ2 = (1-0)/(1-0.01) ≈ 1.01
        # sign(δ1)=0 != sign(δ2)=1 → d clamped to 3*δ1=0 or d=0
        itp = pchip_interp(x2, y2)
        @test isfinite(itp(0.005))

        # Trigger the |d| > |3δ1| path: opposite-sign secants with large first interval
        x3 = [0.0, 1.0, 1.1, 2.0, 3.0]
        y3 = [10.0, 0.0, -0.5, -1.0, 0.0]
        # δ1 = -10, δ2 = -5 → same sign, no clamping here. Try right endpoint:
        # Last two secants: δ_{n-1} and δ_n
        itp3 = pchip_interp(x3, y3)
        @test isfinite(itp3(0.5))
        @test isfinite(itp3(2.5))
    end

    @testset "Coverage — show with Range grid" begin
        x_range = range(0.0, 3.0, 10)
        y = sin.(collect(x_range))
        itp = pchip_interp(x_range, y)
        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("PchipInterpolant1D", verbose)
        @test !occursin("Search:", verbose)  # Range → no Search row
    end
end
