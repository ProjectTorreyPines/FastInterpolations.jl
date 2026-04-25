# ALLOC_THRESHOLD is defined in runtests.jl

@testitem "Cardinal 1D" setup = [AllocConstants] begin

    # Cubic polynomial for accuracy reference
    p(x) = 2x^3 - x^2 + 3x - 1
    dp(x) = 6x^2 - 2x + 3

    @testset "CatmullRom (tension=0) — oneshot scalar" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            val = cardinal_interp(x, y, xq)
            @test val ≈ p(xq) atol = 0.1  # O(h²) accuracy for central FD slopes
        end
    end

    @testset "CatmullRom — oneshot vector" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        result = cardinal_interp(x, y, xq)
        @test result ≈ p.(xq) atol = 0.1
    end

    @testset "CatmullRom — in-place" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        out = zeros(length(xq))
        cardinal_interp!(out, x, y, xq)
        @test out ≈ p.(xq) atol = 0.1
    end

    @testset "CatmullRom — callable interpolant" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)

        itp = cardinal_interp(x, y)
        @test itp isa CardinalInterpolant1D

        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            @test itp(xq) ≈ p(xq) atol = 0.1
        end

        # Vector + in-place
        @test itp(xq_pts) ≈ p.(xq_pts) atol = 0.1
        out = zeros(length(xq_pts))
        itp(out, xq_pts)
        @test out ≈ p.(xq_pts) atol = 0.1
    end

    @testset "Tension parameter" begin
        x = collect(range(0.0, 5.0, 20))
        y = sin.(x)

        # tension=0 → CatmullRom (default)
        val_t0 = cardinal_interp(x, y, 2.5; tension = 0.0)

        # tension=0.5 → tighter curve
        val_t05 = cardinal_interp(x, y, 2.5; tension = 0.5)

        # tension=1.0 → zero slopes at knots (smooth S-curves between knots)
        itp_t1 = cardinal_interp(x, y; tension = 1.0)
        # At grid point: exact, derivative should be ~0
        @test itp_t1(x[5]) ≈ y[5] atol = 1.0e-12
        @test itp_t1(x[5]; deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-12

        # All should be finite
        @test isfinite(val_t0)
        @test isfinite(val_t05)

        # tension=0 and tension=0.5 should give different results
        @test val_t0 != val_t05
    end

    @testset "Tension via callable" begin
        x = collect(range(0.0, 5.0, 20))
        y = sin.(x)

        itp = cardinal_interp(x, y; tension = 0.3)
        @test itp isa CardinalInterpolant1D
        @test isfinite(itp(2.5))
    end

    @testset "DerivOp correctness" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = 1.5

        # Value
        @test cardinal_interp(x, y, xq; deriv = EvalValue()) ≈ p(xq) atol = 0.1

        # First derivative
        @test cardinal_interp(x, y, xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 0.5

        # Fourth+ → zero
        @test cardinal_interp(x, y, xq; deriv = DerivOp(4)) ≈ 0.0 atol = 1.0e-14
    end

    @testset "Extrapolation modes" begin
        x = collect(range(0.0, 3.0, 20))
        y = sin.(x)

        # NoExtrap
        @test_throws DomainError cardinal_interp(x, y, -0.1)
        @test_throws DomainError cardinal_interp(x, y, 3.1)

        # ClampExtrap
        @test cardinal_interp(x, y, -0.5; extrap = ClampExtrap()) ≈ y[1]
        @test cardinal_interp(x, y, 3.5; extrap = ClampExtrap()) ≈ y[end]

        # ExtendExtrap
        @test isfinite(cardinal_interp(x, y, 3.2; extrap = ExtendExtrap()))

        # FillExtrap
        @test cardinal_interp(x, y, -0.5; extrap = FillExtrap(999.0)) ≈ 999.0
        @test cardinal_interp(x, y, 3.5; extrap = FillExtrap(NaN)) |> isnan
        # FillExtrap derivative at OOB → zero
        @test cardinal_interp(x, y, -0.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14
        @test cardinal_interp(x, y, 3.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14

        # WrapExtrap
        @test isfinite(cardinal_interp(x, y, 3.2; extrap = WrapExtrap()))
        # WrapExtrap — vector fast-path includes x_max (no unnecessary wrap overhead)
        xq_boundary = [x[end]]
        out_boundary = similar(xq_boundary)
        cardinal_interp!(out_boundary, x, y, xq_boundary; extrap = WrapExtrap())
        @test isfinite(out_boundary[1])

        # Callable with extrap
        itp = cardinal_interp(x, y; extrap = ClampExtrap())
        @test itp(-0.5) ≈ y[1]
    end

    @testset "Type stability — @inferred" begin
        x = collect(range(0.0, 3.0, 10))
        y = sin.(x)

        @test @inferred(cardinal_interp(x, y, 1.5)) isa Float64
        @test @inferred(cardinal_interp(x, y, [0.5, 1.5, 2.5])) isa Vector{Float64}

        itp = cardinal_interp(x, y)
        @test @inferred(itp(1.5)) isa Float64
    end

    @testset "Zero allocation" begin
        x = collect(range(0.0, 3.0, 50))
        y = sin.(x)

        # Scalar oneshot
        function test_cardinal_scalar_alloc(x, y, xq)
            cardinal_interp(x, y, xq)
            return @allocated cardinal_interp(x, y, xq)
        end
        @test test_cardinal_scalar_alloc(x, y, 1.5) <= ALLOC_THRESHOLD

        # Callable scalar
        itp = cardinal_interp(x, y)
        function test_cardinal_callable_alloc(itp, xq)
            itp(xq)
            return @allocated itp(xq)
        end
        @test test_cardinal_callable_alloc(itp, 1.5) == 0
    end

    @testset "2-point grid" begin
        x = [0.0, 1.0]
        y = [1.0, 3.0]

        # 2-point → linear (slopes = secant * (1 - tension))
        @test cardinal_interp(x, y, 0.5) ≈ 2.0 atol = 1.0e-12
    end

    @testset "Float32" begin
        x32 = collect(range(0.0f0, 3.0f0, 20))
        y32 = sin.(x32)

        val = cardinal_interp(x32, y32, 1.5f0)
        @test val isa Float32
        @test isfinite(val)
    end

    @testset "Integer promotion" begin
        x_int = 0:5
        y_int = [0, 1, 4, 9, 16, 25]
        val = cardinal_interp(collect(x_int), y_int, 2.5)
        @test val isa Float64
        @test isfinite(val)
    end

    @testset "Broadcast" begin
        x = collect(range(0.0, 3.0, 20))
        y = sin.(x)
        itp = cardinal_interp(x, y)
        @test all(isfinite, itp.([0.5, 1.5, 2.5]))
    end

    @testset "Search override" begin
        x = collect(range(0.0, 3.0, 50))
        y = p.(x)
        xq = 1.5

        @test cardinal_interp(x, y, xq; search = BinarySearch()) ≈ p(xq) atol = 0.1
        itp = cardinal_interp(x, y)
        @test itp(xq; search = BinarySearch()) ≈ p(xq) atol = 0.1
    end

    @testset "Show methods" begin
        x = collect(range(0.0, 3.0, 10))
        y = sin.(x)
        itp = cardinal_interp(x, y)

        compact = sprint(show, itp)
        @test occursin("CardinalInterpolant1D", compact)
        @test occursin("cardinal", compact)

        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("CardinalInterpolant1D", verbose)
        @test occursin("cardinal spline", verbose)
        @test occursin("CatmullRom", verbose)

        # Tension stored and shown
        @test itp.tension == 0.0
        itp2 = cardinal_interp(x, y; tension = 0.5)
        @test itp2.tension == 0.5
        verbose2 = sprint(show, MIME"text/plain"(), itp2)
        @test occursin("tension=0.5", verbose2)
    end

    @testset "In-place hint forwarding and zero-alloc" begin
        x = collect(range(0.0, 4.0, 50))
        y = sin.(x)
        xq = collect(range(0.1, 3.9, 20))
        output = Vector{Float64}(undef, length(xq))

        # hint gets updated through in-place path
        hint = Ref(1)
        cardinal_interp!(output, x, y, xq; hint)
        @test hint[] > 1

        # Verify correctness with hint
        ref = cardinal_interp(x, y, xq)
        @test output ≈ ref

        # Zero-allocation in-place (function barrier)
        function test_cardinal_inplace_alloc(output, x, y, xq)
            hint_local = Ref(1)
            cardinal_interp!(output, x, y, xq; hint = hint_local)
            return @allocated cardinal_interp!(output, x, y, xq; hint = hint_local)
        end
        @test test_cardinal_inplace_alloc(output, x, y, xq) <= ALLOC_THRESHOLD
    end

    @testset "Coverage — Range in-place disambiguation" begin
        x_range = range(0.0, 3.0, 20)
        y = sin.(collect(x_range))
        xq = collect(range(0.1, 2.9, 10))
        out = similar(xq)
        cardinal_interp!(out, x_range, y, xq)
        @test out ≈ cardinal_interp(collect(x_range), y, xq) atol = 1.0e-12
    end

    @testset "Coverage — vector tension variation" begin
        x = collect(range(0.0, 3.0, 20))
        y = sin.(x)
        xq = [0.5, 1.0, 2.0]

        # Verify tension affects vector output
        v_t0 = cardinal_interp(x, y, xq; tension = 0.0)
        v_t05 = cardinal_interp(x, y, xq; tension = 0.5)
        @test v_t0 != v_t05  # different tension → different results

        # In-place with tension
        out = similar(xq)
        cardinal_interp!(out, x, y, xq; tension = 0.3)
        @test all(isfinite, out)
    end

    @testset "Coverage — generic vector-allocating wrapper (Integer inputs)" begin
        x_int = collect(0:9)
        y_int = x_int .^ 2
        xq_int = [2, 4, 6]
        result = cardinal_interp(x_int, y_int, xq_int)
        @test length(result) == 3
        @test all(isfinite, result)
        @test eltype(result) == Float64
    end

    @testset "Coverage — generic scalar wrapper (Integer inputs)" begin
        x_int = collect(0:9)
        y_int = x_int .^ 2
        @test isfinite(cardinal_interp(x_int, y_int, 3))
    end

    @testset "Coverage — generic in-place pass-through (Integer inputs)" begin
        x_int = collect(0:9)
        y_int = x_int .^ 2
        xq_int = [2, 4, 6]
        out = Vector{Float64}(undef, length(xq_int))
        cardinal_interp!(out, x_int, y_int, xq_int)
        @test all(isfinite, out)
    end

    @testset "Coverage — WrapExtrap vector path" begin
        x = collect(range(0.0, 2π, 20))
        y = sin.(x)
        xq_inner = collect(range(0.1, 2π - 0.1, 10))
        out = similar(xq_inner)
        cardinal_interp!(out, x, y, xq_inner; extrap = WrapExtrap())
        @test all(isfinite, out)
    end

    @testset "Coverage — show with Range grid" begin
        x_range = range(0.0, 3.0, 10)
        y = sin.(collect(x_range))
        itp = cardinal_interp(x_range, y)
        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("CardinalInterpolant1D", verbose)
        @test !occursin("Search:", verbose)
    end

    @testset "Coverage — n=2 minimum grid" begin
        x = [0.0, 1.0]
        y = [1.0, 3.0]
        @test isfinite(cardinal_interp(x, y, 0.5))
    end
end
