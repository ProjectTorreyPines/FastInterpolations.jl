# Tests for PeriodicBC on Cardinal ND forward path (OnTheFly only).
# Mirrors test_pchip_periodic_nd.jl with extra `tension` coverage.

@testitem "Cardinal PeriodicBC ND forward" setup = [AllocConstants] begin
    f(x, y) = sin(2π * x) * cos(2π * y)

    @testset "NoBC default is no-op" begin
        n = 11
        x = collect(range(0.0, 1.0, length = n))
        y = collect(range(0.0, 1.0, length = n))
        data = [f(xi, yj) for xi in x, yj in y]

        v_default = cardinal_interp((x, y), data, (0.4, 0.6))
        v_nobc = cardinal_interp((x, y), data, (0.4, 0.6); bc = NoBC())
        @test v_default === v_nobc
    end

    @testset "Homogeneous Cardinal × Cardinal — bc broadcast (exclusive)" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        for tens in (0.0, 0.5)
            v = cardinal_interp((x, y), data, (0.5, 0.5); bc = bc, tension = tens)
            @test isapprox(v, f(0.5, 0.5); atol = 1.0e-2)
            ε = 1.0e-7
            v_left = cardinal_interp((x, y), data, (1.0 - ε, 0.5); bc = bc, tension = tens)
            v_right = cardinal_interp((x, y), data, (0.0 + ε, 0.5); bc = bc, tension = tens)
            @test abs(v_left - v_right) < 1.0e-3
        end
    end

    @testset "Inclusive endpoint" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n))
        y = collect(range(0.0, 1.0, length = n))
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :inclusive)

        v = cardinal_interp((x, y), data, (0.5, 0.5); bc = bc, tension = 0.0)
        @test isapprox(v, f(0.5, 0.5); atol = 1.0e-3)
        ε = 1.0e-7
        @test abs(
            cardinal_interp((x, y), data, (1.0 - ε, 0.5); bc = bc) -
                cardinal_interp((x, y), data, (0.0 + ε, 0.5); bc = bc),
        ) < 1.0e-3
    end

    @testset "Heterogeneous: CardinalInterp(tension, bc) × LinearInterp()" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        v = interp((x, y), data, (0.5, 0.5); method = (CardinalInterp(0.3, bc), LinearInterp()))
        @test isapprox(v, f(0.5, 0.5); atol = 1.0e-2)
    end

    @testset "Persistent ND interpolant + vector batch" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = cardinal_interp((x, y), data; bc = bc, tension = 0.5)
        @test isapprox(itp((0.5, 0.5)), f(0.5, 0.5); atol = 1.0e-2)

        queries = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.85)]
        v_alloc = cardinal_interp((x, y), data, queries; bc = bc)
        out = similar(v_alloc)
        cardinal_interp!(out, (x, y), data, queries; bc = bc)
        @test out == v_alloc
    end

    @testset "Type stability (@inferred)" begin
        function _inferred_cardinal_periodic_nd()
            n = 21
            x = collect(range(0.0, 1.0, length = n + 1))[1:n]
            y = collect(range(0.0, 1.0, length = n + 1))[1:n]
            data = [f(xi, yj) for xi in x, yj in y]
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            return @inferred cardinal_interp((x, y), data, (0.5, 0.5); bc = bc, tension = 0.3)
        end
        v = _inferred_cardinal_periodic_nd()
        @test v isa Float64
    end

    @testset "Zero-alloc scalar oneshot (function barrier)" begin
        function _alloc_check()
            n = 21
            x = collect(range(0.0, 1.0, length = n + 1))[1:n]
            y = collect(range(0.0, 1.0, length = n + 1))[1:n]
            data = [f(xi, yj) for xi in x, yj in y]
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            cardinal_interp((x, y), data, (0.5, 0.5); bc = bc, tension = 0.3)   # warmup
            return @allocated cardinal_interp((x, y), data, (0.5, 0.5); bc = bc, tension = 0.3)
        end
        @test _alloc_check() <= ALLOC_THRESHOLD
    end
end
