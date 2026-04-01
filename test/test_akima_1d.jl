# ALLOC_THRESHOLD is defined in runtests.jl

@testset "Akima 1D" begin

    p(x) = 2x^3 - x^2 + 3x - 1
    dp(x) = 6x^2 - 2x + 3

    @testset "Basic correctness — oneshot scalar" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            val = akima_interp(x, y, xq)
            @test val ≈ p(xq) atol = 0.05
        end
    end

    @testset "Basic correctness — oneshot vector" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        result = akima_interp(x, y, xq)
        @test result ≈ p.(xq) atol = 0.05
    end

    @testset "Basic correctness — in-place" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        out = zeros(length(xq))
        akima_interp!(out, x, y, xq)
        @test out ≈ p.(xq) atol = 0.05
    end

    @testset "Basic correctness — callable interpolant" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)

        itp = akima_interp(x, y)
        @test itp isa AkimaInterpolant1D

        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            @test itp(xq) ≈ p(xq) atol = 0.05
        end

        @test itp(xq_pts) ≈ p.(xq_pts) atol = 0.05

        out = zeros(length(xq_pts))
        itp(out, xq_pts)
        @test out ≈ p.(xq_pts) atol = 0.05
    end

    @testset "Outlier robustness" begin
        # Data with one outlier at x=2
        x = collect(range(0.0, 5.0, 6))
        y_clean = sin.(x)
        y_outlier = copy(y_clean)
        y_outlier[3] = 10.0  # huge outlier at x=2

        itp_clean = akima_interp(x, y_clean)
        itp_outlier = akima_interp(x, y_outlier)

        # Far from the outlier, Akima should give similar results
        # Query at x=4.5 (far from outlier at x=2)
        val_clean = itp_clean(4.5)
        val_outlier = itp_outlier(4.5)
        @test abs(val_clean - val_outlier) < 0.5  # Akima limits outlier spread
    end

    @testset "DerivOp correctness" begin
        x = collect(range(0.0, 3.0, 30))
        y = p.(x)
        xq = 1.5

        @test akima_interp(x, y, xq; deriv = EvalValue()) ≈ p(xq) atol = 0.05
        @test akima_interp(x, y, xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 0.1
        @test akima_interp(x, y, xq; deriv = DerivOp(4)) ≈ 0.0 atol = 1.0e-14

        itp = akima_interp(x, y)
        @test itp(xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 0.1
    end

    @testset "Extrapolation modes" begin
        x = collect(range(0.0, 3.0, 20))
        y = sin.(x)

        @test_throws DomainError akima_interp(x, y, -0.1)
        @test_throws DomainError akima_interp(x, y, 3.1)

        @test akima_interp(x, y, -0.5; extrap = ClampExtrap()) ≈ y[1]
        @test akima_interp(x, y, 3.5; extrap = ClampExtrap()) ≈ y[end]

        @test isfinite(akima_interp(x, y, 3.2; extrap = ExtendExtrap()))
        @test akima_interp(x, y, -0.5; extrap = FillExtrap(999.0)) ≈ 999.0
        # FillExtrap derivative at OOB → zero
        @test akima_interp(x, y, -0.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14
        @test akima_interp(x, y, 3.5; extrap = FillExtrap(999.0), deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14
        @test isfinite(akima_interp(x, y, 3.2; extrap = WrapExtrap()))
        # WrapExtrap — vector fast-path includes x_max (no unnecessary wrap overhead)
        xq_boundary = [x[end]]
        out_boundary = similar(xq_boundary)
        akima_interp!(out_boundary, x, y, xq_boundary; extrap = WrapExtrap())
        @test isfinite(out_boundary[1])

        itp = akima_interp(x, y; extrap = ClampExtrap())
        @test itp(-0.5) ≈ y[1]
    end

    @testset "Type stability — @inferred" begin
        x = collect(range(0.0, 3.0, 10))
        y = sin.(x)

        @test @inferred(akima_interp(x, y, 1.5)) isa Float64
        @test @inferred(akima_interp(x, y, [0.5, 1.5, 2.5])) isa Vector{Float64}

        itp = akima_interp(x, y)
        @test @inferred(itp(1.5)) isa Float64
    end

    @testset "Zero allocation" begin
        x = collect(range(0.0, 3.0, 50))
        y = sin.(x)

        function test_akima_scalar_alloc(x, y, xq)
            akima_interp(x, y, xq)
            return @allocated akima_interp(x, y, xq)
        end
        @test test_akima_scalar_alloc(x, y, 1.5) <= ALLOC_THRESHOLD

        itp = akima_interp(x, y)
        function test_akima_callable_alloc(itp, xq)
            itp(xq)
            return @allocated itp(xq)
        end
        @test test_akima_callable_alloc(itp, 1.5) == 0
    end

    @testset "Small grids" begin
        # 2-point → linear
        x2 = [0.0, 1.0]
        y2 = [1.0, 3.0]
        @test akima_interp(x2, y2, 0.5) ≈ 2.0 atol = 1.0e-12

        # 3-point → simple average at interior
        x3 = [0.0, 1.0, 3.0]
        y3 = [1.0, 2.0, 0.5]
        @test isfinite(akima_interp(x3, y3, 1.5))

        # 4-point → full algorithm without loop body
        x4 = [0.0, 1.0, 2.0, 3.0]
        y4 = [0.0, 1.0, 0.5, 2.0]
        @test isfinite(akima_interp(x4, y4, 1.5))
    end

    @testset "Flat data — zero slopes" begin
        x = collect(range(0.0, 5.0, 10))
        y = fill(3.0, 10)

        itp = akima_interp(x, y)
        @test itp(2.5) ≈ 3.0 atol = 1.0e-14
        @test itp(2.5; deriv = DerivOp(1)) ≈ 0.0 atol = 1.0e-14
    end

    @testset "Integer promotion" begin
        x_int = 0:5
        y_int = [0, 1, 4, 9, 16, 25]
        val = akima_interp(collect(x_int), y_int, 2.5)
        @test val isa Float64
        @test isfinite(val)
    end

    @testset "Float32" begin
        x32 = collect(range(0.0f0, 3.0f0, 20))
        y32 = sin.(x32)

        val = akima_interp(x32, y32, 1.5f0)
        @test val isa Float32
        @test isfinite(val)

        itp32 = akima_interp(x32, y32)
        @test itp32(1.5f0) isa Float32
    end

    @testset "Broadcast" begin
        x = collect(range(0.0, 3.0, 20))
        y = sin.(x)
        itp = akima_interp(x, y)
        @test all(isfinite, itp.([0.5, 1.5, 2.5]))
    end

    @testset "Search override" begin
        x = collect(range(0.0, 3.0, 50))
        y = p.(x)
        xq = 1.5

        @test akima_interp(x, y, xq; search = BinarySearch()) ≈ p(xq) atol = 0.05
        itp = akima_interp(x, y)
        @test itp(xq; search = BinarySearch()) ≈ p(xq) atol = 0.05
    end

    @testset "Akima vs cubic spline" begin
        x = collect(range(0.0, 2π, 20))
        y = sin.(x)
        xq = [0.5, 1.5, 2.5, 3.5, 4.5]

        akima_vals = akima_interp(x, y, xq)
        spline_vals = cubic_interp(x, y, xq)

        @test akima_vals ≈ sin.(xq) atol = 0.05
        @test spline_vals ≈ sin.(xq) atol = 0.05
    end

    @testset "Show methods" begin
        x = collect(range(0.0, 3.0, 10))
        y = sin.(x)
        itp = akima_interp(x, y)

        compact = sprint(show, itp)
        @test occursin("AkimaInterpolant1D", compact)
        @test occursin("outlier-robust", compact)

        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("AkimaInterpolant1D", verbose)
        @test occursin("Akima", verbose)
    end

    @testset "In-place hint forwarding and zero-alloc" begin
        x = collect(range(0.0, 4.0, 50))
        y = sin.(x)
        xq = collect(range(0.1, 3.9, 20))
        output = Vector{Float64}(undef, length(xq))

        # hint gets updated through in-place path
        hint = Ref(1)
        akima_interp!(output, x, y, xq; hint)
        @test hint[] > 1

        # Verify correctness with hint
        ref = akima_interp(x, y, xq)
        @test output ≈ ref

        # Zero-allocation in-place (function barrier)
        function test_akima_inplace_alloc(output, x, y, xq)
            hint_local = Ref(1)
            akima_interp!(output, x, y, xq; hint = hint_local)
            return @allocated akima_interp!(output, x, y, xq; hint = hint_local)
        end
        @test test_akima_inplace_alloc(output, x, y, xq) <= ALLOC_THRESHOLD
    end

    @testset "Coverage — n=2 minimum grid" begin
        x = [0.0, 1.0]
        y = [1.0, 3.0]
        # n=2: single secant, endpoints get that secant as slope
        @test akima_interp(x, y, 0.5) ≈ 2.0 atol = 0.1
        itp = akima_interp(x, y)
        @test isfinite(itp(0.5))
    end

    @testset "Coverage — n=3 special case" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 0.0]
        # n=3: arithmetic mean for interior, one-sided for endpoints
        @test isfinite(akima_interp(x, y, 0.5))
        @test isfinite(akima_interp(x, y, 1.5))
        # Midpoint slope should be (m1+m2)/2 = (1 + (-1))/2 = 0
        @test akima_interp(x, y, 1.0) ≈ y[2] atol = 1.0e-14
    end

    @testset "Coverage — wsum==0 fallback (equal secant differences)" begin
        # Equally spaced linear data: all secants identical → w1=w2=0 → fallback
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        # All secants = 1.0, differences = 0 → wsum = 0 → arithmetic mean fallback
        @test akima_interp(x, y, 2.5) ≈ 2.5 atol = 1.0e-12
        itp = akima_interp(x, y)
        @test itp(1.5) ≈ 1.5 atol = 1.0e-12
    end

    @testset "Coverage — Range in-place disambiguation" begin
        x_range = range(0.0, 3.0, 20)
        y = sin.(collect(x_range))
        xq = collect(range(0.1, 2.9, 10))
        out = similar(xq)
        akima_interp!(out, x_range, y, xq)
        @test out ≈ akima_interp(collect(x_range), y, xq) atol = 1.0e-12
    end

    @testset "Coverage — output eltype validation error" begin
        x_int = collect(0:9)
        y_int = x_int .^ 2
        xq_int = [2, 4, 6]
        out_narrow = Vector{Float32}(undef, length(xq_int))
        @test_throws ArgumentError akima_interp!(out_narrow, x_int, y_int, xq_int)
    end

    @testset "Coverage — WrapExtrap vector path" begin
        x = collect(range(0.0, 2π, 20))
        y = sin.(x)
        xq_inner = collect(range(0.1, 2π - 0.1, 10))
        out = similar(xq_inner)
        akima_interp!(out, x, y, xq_inner; extrap = WrapExtrap())
        @test all(isfinite, out)
    end

    @testset "Coverage — show with Range grid" begin
        x_range = range(0.0, 3.0, 10)
        y = sin.(collect(x_range))
        itp = akima_interp(x_range, y)
        verbose = sprint(show, MIME"text/plain"(), itp)
        @test occursin("AkimaInterpolant1D", verbose)
        @test !occursin("Search:", verbose)
    end

    @testset "Coverage — n=4 boundary (loop edge case)" begin
        # n=4: interior loop runs k=3:2 (empty), boundary code handles all slopes
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 0.0, 1.0]
        @test isfinite(akima_interp(x, y, 0.5))
        @test isfinite(akima_interp(x, y, 1.5))
        @test isfinite(akima_interp(x, y, 2.5))
    end
end
