# Duck-typing tests for non-promotable Tv (e.g., SVector) crossed with non-
# promotable Tq (e.g., ForwardDiff.Dual).
#
# These cases are the intersection where `_output_eltype(Tv, Tg, Tq)` falls
# back (its `promote_type` returns a non-concrete typejoin). Tests pin the
# expected return types and container concreteness across both 1D and ND
# entries — scalar callable, ForwardDiff.derivative form, allocating batch,
# and the Constant raw-eltype contract.

@testitem "Duck Tv × Dual Tq — 1D scalar query widens correctly" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]
    xq_d = ForwardDiff.Dual{Nothing}(2.5, 1.0)
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Arithmetic kernels widen SVector{N, Float64} → SVector{N, Dual}" begin
        @test linear_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test cubic_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test quadratic_interp(x, y_sv)(xq_d) isa SVector{3, D}
        @test hermite_interp(x, y_sv, dy_sv)(xq_d) isa SVector{3, D}
    end

    @testset "Constant kernel widens SVector{N, Float64} → SVector{N, Dual}" begin
        @test constant_interp(x, y_sv)(xq_d) isa SVector{3, D}
    end
end

@testitem "Duck Tv × Dual Tq — ForwardDiff.derivative of indexed component" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    dy_sv = [SA[1.0, 2.0, 3.0] for _ in 1:10]

    @testset "linear_interp" begin
        itp = linear_interp(x, y_sv)
        d = ForwardDiff.derivative(t -> itp(t)[1], 2.5)
        @test d isa Float64
        @test isfinite(d)
    end

    @testset "cubic_interp" begin
        itp = cubic_interp(x, y_sv)
        d = ForwardDiff.derivative(t -> itp(t)[1], 2.5)
        @test d isa Float64
        @test isfinite(d)
    end

    @testset "quadratic_interp" begin
        itp = quadratic_interp(x, y_sv)
        d = ForwardDiff.derivative(t -> itp(t)[1], 2.5)
        @test d isa Float64
        @test isfinite(d)
    end

    @testset "hermite_interp" begin
        itp = hermite_interp(x, y_sv, dy_sv)
        d = ForwardDiff.derivative(t -> itp(t)[1], 2.5)
        @test d isa Float64
        @test isfinite(d)
    end

    @testset "constant_interp returns zero derivative" begin
        itp = constant_interp(x, y_sv)
        d = ForwardDiff.derivative(t -> itp(t)[1], 2.5)
        @test d isa Float64
        @test d == 0.0
    end
end

@testitem "Duck Tv × Dual Tq — Constant 1D batch widens to SVector{N, Dual}" begin
    using StaticArrays, ForwardDiff

    x = collect(1.0:10.0)
    y_sv = [SA[Float64(i), 2.0i, 3.0i] for i in 1:10]
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @testset "Persistent batch" begin
        out = constant_interp(x, y_sv)(xq_dv)
        @test out isa Vector{SVector{3, D}}
        @test eltype(out) === SVector{3, D}
        @test length(out) == 5
    end

    @testset "Oneshot 3-arg batch" begin
        out = constant_interp(x, y_sv, xq_dv)
        @test out isa Vector{SVector{3, D}}
        @test eltype(out) === SVector{3, D}
    end
end

@testitem "Duck Tv × Dual Tq — Constant ND batch widens to SVector{N, Dual}" begin
    using StaticArrays, ForwardDiff

    xg = collect(1.0:5.0)
    yg = collect(1.0:5.0)
    data_sv = [SA[Float64(i + j), 2.0(i + j), 3.0(i + j)] for i in 1:5, j in 1:5]
    q_dv = [
        (ForwardDiff.Dual{Nothing}(2.5 + 0.1i, 1.0), ForwardDiff.Dual{Nothing}(3.5 + 0.1i, 0.0))
            for i in 1:5
    ]
    D = ForwardDiff.Dual{Nothing, Float64, 1}

    out = constant_interp((xg, yg), data_sv)(q_dv)
    @test out isa Vector{SVector{3, D}}
    @test eltype(out) === SVector{3, D}
    @test length(out) == 5
end

@testitem "Plain Float64 y + Vector{Dual} batch preserves Dual eltype (non-regression)" begin
    using ForwardDiff

    x = collect(1.0:10.0)
    y = sin.(x)
    xq_dv = [ForwardDiff.Dual{Nothing}(2.0 + 0.1i, 1.0) for i in 1:5]

    D = ForwardDiff.Dual{Nothing, Float64, 1}

    @test linear_interp(x, y)(xq_dv) isa Vector{D}
    @test cubic_interp(x, y)(xq_dv) isa Vector{D}
    @test quadratic_interp(x, y)(xq_dv) isa Vector{D}
    @test hermite_interp(x, y, cos.(x))(xq_dv) isa Vector{D}
    @test constant_interp(x, y)(xq_dv) isa Vector{D}
end

@testitem "Constant raw-eltype contract: Int values stay Int (non-regression)" begin
    x = collect(1:10)
    y = collect(10:10:100)
    @test constant_interp(x, y)(3) isa Int
    @test constant_interp(x, y)([2, 5, 8]) isa Vector{Int}
end
