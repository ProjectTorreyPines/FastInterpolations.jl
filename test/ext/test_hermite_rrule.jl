# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              HERMITE FAMILY rrule TESTS (ChainRulesCore)                 ║
# ║  Tests rrule wiring for cardinal, pchip, akima, and Hermite paths       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

using ChainRulesCore
using ChainRulesCore: rrule, NoTangent, ZeroTangent, Tangent, unthunk

@testset "Hermite family — rrule wiring" begin
    # ════════════════════════════════════════════════════════════════════════
    # Cardinal (generic _InterpMethod path)
    # ════════════════════════════════════════════════════════════════════════

    @testset "cardinal_interp rrule (generic path)" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(cardinal_interp, x, y, xq)
        @test val ≈ cardinal_interp(x, y, xq)
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[3] isa Vector
        @test length(grads[3]) == length(y)
        # Verify ∂L/∂y matches adjoint operator
        adj = cardinal_adjoint(x, xq)
        @test grads[3] ≈ adj(Δy)
    end

    @testset "cardinal rrule scalar xq" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        val, pb = rrule(cardinal_interp, x, y, 0.5)
        @test val ≈ cardinal_interp(x, y, 0.5)
        grads = pb(1.0)
        @test grads[3] isa Vector
        @test length(grads[3]) == length(y)
    end

    @testset "cardinal rrule with tension" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(cardinal_interp, x, y, xq; tension = 0.5)
        @test val ≈ cardinal_interp(x, y, xq; tension = 0.5)
        Δy = randn(2)
        grads = pb(Δy)
        adj = cardinal_adjoint(x, xq; tension = 0.5)
        @test grads[3] ≈ adj(Δy)
    end

    @testset "cardinal rrule ∂xq (EvalValue)" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(cardinal_interp, x, y, xq)
        Δy = randn(2)
        grads = pb(Δy)
        # ∂xq should be a vector of same length
        @test grads[4] isa AbstractVector
        @test length(grads[4]) == length(xq)
    end

    @testset "cardinal rrule AbstractZero passthrough" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        _, pb = rrule(cardinal_interp, x, y, xq)
        grads = pb(ZeroTangent())
        @test grads[3] isa ZeroTangent
    end

    # ════════════════════════════════════════════════════════════════════════
    # PCHIP (specialized rrule)
    # ════════════════════════════════════════════════════════════════════════

    @testset "pchip_interp rrule (specialized path)" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(pchip_interp, x, y, xq)
        @test val ≈ pchip_interp(x, y, xq)
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[3] isa Vector
        @test length(grads[3]) == length(y)
        adj = pchip_adjoint(x, y, xq)
        @test grads[3] ≈ adj(Δy)
    end

    @testset "pchip rrule scalar xq" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        val, pb = rrule(pchip_interp, x, y, 0.5)
        @test val ≈ pchip_interp(x, y, 0.5)
        grads = pb(1.0)
        @test grads[3] isa Vector
    end

    @testset "pchip rrule ∂xq (EvalValue)" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        _, pb = rrule(pchip_interp, x, y, xq)
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[4] isa AbstractVector
        @test length(grads[4]) == length(xq)
    end

    @testset "pchip rrule with deriv=EvalDeriv1" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(pchip_interp, x, y, xq; deriv = DerivOp(1))
        @test val ≈ pchip_interp(x, y, xq; deriv = DerivOp(1))
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[3] isa Vector
        # ∂xq should be NoTangent for non-EvalValue deriv
        @test grads[4] isa NoTangent
    end

    @testset "pchip rrule AbstractZero passthrough" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        _, pb = rrule(pchip_interp, x, y, xq)
        grads = pb(ZeroTangent())
        @test grads[3] isa ZeroTangent
    end

    # ════════════════════════════════════════════════════════════════════════
    # Akima (specialized rrule)
    # ════════════════════════════════════════════════════════════════════════

    @testset "akima_interp rrule (specialized path)" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(akima_interp, x, y, xq)
        @test val ≈ akima_interp(x, y, xq)
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[3] isa Vector
        @test length(grads[3]) == length(y)
        adj = akima_adjoint(x, y, xq)
        @test grads[3] ≈ adj(Δy)
    end

    @testset "akima rrule scalar xq" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        val, pb = rrule(akima_interp, x, y, 0.5)
        @test val ≈ akima_interp(x, y, 0.5)
        grads = pb(1.0)
        @test grads[3] isa Vector
    end

    @testset "akima rrule ∂xq (EvalValue)" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        _, pb = rrule(akima_interp, x, y, xq)
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[4] isa AbstractVector
        @test length(grads[4]) == length(xq)
    end

    @testset "akima rrule with deriv=EvalDeriv1" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(akima_interp, x, y, xq; deriv = DerivOp(1))
        @test val ≈ akima_interp(x, y, xq; deriv = DerivOp(1))
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[3] isa Vector
        @test grads[4] isa NoTangent
    end

    @testset "akima rrule AbstractZero passthrough" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        xq = [0.3, 0.7]
        _, pb = rrule(akima_interp, x, y, xq)
        grads = pb(ZeroTangent())
        @test grads[3] isa ZeroTangent
    end

    # ════════════════════════════════════════════════════════════════════════
    # Hermite (hermite_interp with separate y, dy args)
    # ════════════════════════════════════════════════════════════════════════
    # rrule returns 5-tuple: (func, x, ∂y, ∂dy, ∂xq)

    @testset "hermite_interp rrule — ∂y and ∂dy" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(hermite_interp, x, y, dy, xq)
        @test val ≈ hermite_interp(x, y, dy, xq)
        Δy = randn(2)
        grads = pb(Δy)
        # grads: (func, x, ∂y, ∂dy, ∂xq)
        @test grads[3] isa Vector     # ∂y
        @test grads[4] isa Vector     # ∂dy
        @test length(grads[3]) == length(y)
        @test length(grads[4]) == length(dy)
    end

    @testset "hermite_interp rrule scalar xq" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        val, pb = rrule(hermite_interp, x, y, dy, 0.5)
        @test val ≈ hermite_interp(x, y, dy, 0.5)
        grads = pb(1.0)
        @test grads[3] isa Vector     # ∂y
        @test grads[4] isa Vector     # ∂dy
    end

    @testset "hermite_interp rrule ∂xq" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        xq = [0.3, 0.7]
        _, pb = rrule(hermite_interp, x, y, dy, xq)
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[5] isa AbstractVector   # ∂xq is now index 5
        @test length(grads[5]) == length(xq)
    end

    @testset "hermite_interp rrule deriv=EvalDeriv1" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        xq = [0.3, 0.7]
        val, pb = rrule(hermite_interp, x, y, dy, xq; deriv = DerivOp(1))
        @test val ≈ hermite_interp(x, y, dy, xq; deriv = DerivOp(1))
        Δy = randn(2)
        grads = pb(Δy)
        @test grads[3] isa Vector     # ∂y
        @test grads[4] isa Vector     # ∂dy
        @test grads[5] isa NoTangent  # ∂xq not available for non-EvalValue
    end

    @testset "hermite_interp AbstractZero passthrough" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        xq = [0.3, 0.7]
        _, pb = rrule(hermite_interp, x, y, dy, xq)
        grads = pb(ZeroTangent())
        @test grads[3] isa ZeroTangent
    end

    @testset "hermite_interp ∂y matches finite-diff" begin
        x = sort(vcat(0.0, rand(18), 1.0))
        y = randn(length(x))
        dy = randn(length(x))
        xq = sort(rand(10))

        _, pb = rrule(hermite_interp, x, y, dy, xq)
        Δy = randn(length(xq))
        ∂y_ad = pb(Δy)[3]

        # Finite-difference: perturb y[k] by ε
        ε = 1.0e-7
        ∂y_fd = similar(y)
        for k in eachindex(y)
            y_plus = copy(y); y_plus[k] += ε
            y_minus = copy(y); y_minus[k] -= ε
            f_plus = hermite_interp(x, y_plus, dy, xq)
            f_minus = hermite_interp(x, y_minus, dy, xq)
            ∂y_fd[k] = sum(Δy .* (f_plus .- f_minus)) / (2ε)
        end
        @test ∂y_ad ≈ ∂y_fd rtol = 1.0e-5
    end

    @testset "hermite_interp ∂dy matches finite-diff" begin
        x = sort(vcat(0.0, rand(18), 1.0))
        y = randn(length(x))
        dy = randn(length(x))
        xq = sort(rand(10))

        _, pb = rrule(hermite_interp, x, y, dy, xq)
        Δy = randn(length(xq))
        ∂dy_ad = pb(Δy)[4]

        # Finite-difference: perturb dy[k] by ε
        ε = 1.0e-7
        ∂dy_fd = similar(dy)
        for k in eachindex(dy)
            dy_plus = copy(dy); dy_plus[k] += ε
            dy_minus = copy(dy); dy_minus[k] -= ε
            f_plus = hermite_interp(x, y, dy_plus, xq)
            f_minus = hermite_interp(x, y, dy_minus, xq)
            ∂dy_fd[k] = sum(Δy .* (f_plus .- f_minus)) / (2ε)
        end
        @test ∂dy_ad ≈ ∂dy_fd rtol = 1.0e-5
    end
end
