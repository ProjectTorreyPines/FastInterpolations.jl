# Tests for PeriodicBC on Akima ND forward path (OnTheFly only).

@testitem "Akima PeriodicBC ND forward" setup = [AllocConstants] begin
    f(x, y) = sin(2π * x) * cos(2π * y)

    @testset "NoBC default is no-op" begin
        n = 11
        x = collect(range(0.0, 1.0, length = n))
        y = collect(range(0.0, 1.0, length = n))
        data = [f(xi, yj) for xi in x, yj in y]

        v_default = akima_interp((x, y), data, (0.4, 0.6))
        v_nobc = akima_interp((x, y), data, (0.4, 0.6); bc = NoBC())
        @test v_default === v_nobc
    end

    @testset "Homogeneous Akima × Akima — bc broadcast (exclusive)" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        v = akima_interp((x, y), data, (0.5, 0.5); bc = bc)
        @test isapprox(v, f(0.5, 0.5); atol = 1.0e-3)
        ε = 1.0e-7
        v_left = akima_interp((x, y), data, (1.0 - ε, 0.5); bc = bc)
        v_right = akima_interp((x, y), data, (0.0 + ε, 0.5); bc = bc)
        @test abs(v_left - v_right) < 1.0e-3
    end

    @testset "Inclusive endpoint" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n))
        y = collect(range(0.0, 1.0, length = n))
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :inclusive)

        v = akima_interp((x, y), data, (0.5, 0.5); bc = bc)
        @test isapprox(v, f(0.5, 0.5); atol = 1.0e-3)
        ε = 1.0e-7
        @test abs(
            akima_interp((x, y), data, (1.0 - ε, 0.5); bc = bc) -
                akima_interp((x, y), data, (0.0 + ε, 0.5); bc = bc),
        ) < 1.0e-3
    end

    @testset "Heterogeneous: AkimaInterp(bc) × LinearInterp()" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        v = interp((x, y), data, (0.5, 0.5); method = (AkimaInterp(bc), LinearInterp()))
        @test isapprox(v, f(0.5, 0.5); atol = 1.0e-2)
    end

    @testset "Persistent ND interpolant + vector batch" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = akima_interp((x, y), data; bc = bc)
        @test isapprox(itp((0.5, 0.5)), f(0.5, 0.5); atol = 1.0e-3)

        queries = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.85)]
        v_alloc = akima_interp((x, y), data, queries; bc = bc)
        out = similar(v_alloc)
        akima_interp!(out, (x, y), data, queries; bc = bc)
        @test out == v_alloc
    end

    @testset "Type stability (@inferred)" begin
        function _inferred_akima_periodic_nd()
            n = 21
            x = collect(range(0.0, 1.0, length = n + 1))[1:n]
            y = collect(range(0.0, 1.0, length = n + 1))[1:n]
            data = [f(xi, yj) for xi in x, yj in y]
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            return @inferred akima_interp((x, y), data, (0.5, 0.5); bc = bc)
        end
        v = _inferred_akima_periodic_nd()
        @test v isa Float64
    end

    @testset "Zero-alloc scalar oneshot (function barrier)" begin
        function _alloc_check()
            n = 21
            x = collect(range(0.0, 1.0, length = n + 1))[1:n]
            y = collect(range(0.0, 1.0, length = n + 1))[1:n]
            data = [f(xi, yj) for xi in x, yj in y]
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            akima_interp((x, y), data, (0.5, 0.5); bc = bc)   # warmup
            return @allocated akima_interp((x, y), data, (0.5, 0.5); bc = bc)
        end
        @test _alloc_check() <= ALLOC_THRESHOLD
    end
end
