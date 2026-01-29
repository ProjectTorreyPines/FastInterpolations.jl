using Test
using FastInterpolations
using Random

const FI = FastInterpolations
const LA = FI.LinearAlgebra

@testset "Tridiagonal NoPivot _ldiv_tridiagonal_nopiv!" begin
    function _x_uniform(::Type{T}, n::Int) where {T<:AbstractFloat}
        return collect(range(T(0), T(1), n))
    end

    function _x_nonuniform(::Type{T}, n::Int; seed::Int=0) where {T<:AbstractFloat}
        # Random positive steps, normalized to unit length, then cumulative sum.
        rng = Random.MersenneTwister(seed)
        steps = rand(rng, T, n - 1) .+ T(0.1)
        steps ./= sum(steps)
        x = Vector{T}(undef, n)
        x[1] = zero(T)
        @inbounds for i in 1:(n - 1)
            x[i + 1] = x[i] + steps[i]
        end
        x[end] = one(T)
        return x
    end

    function _x_strongly_nonuniform(::Type{T}, n::Int; strength::Symbol=:strong) where {T<:AbstractFloat}
        # Construct strictly-increasing x with a large step ratio while keeping Float32 representable.
        n_steps = n - 1
        steps = Vector{T}(undef, n_steps)
        if T === Float32
            # Safe for Float32: split steps into tiny and large blocks (ratio ~ O(1e2-1e3)).
            k = n_steps ÷ 2
            small = strength === :strong ? T(1e-4) : T(1e-3)
            @inbounds for i in 1:k
                steps[i] = small
            end
            remaining = one(T) - small * k
            big = remaining / (n_steps - k)
            @inbounds for i in (k + 1):n_steps
                steps[i] = big
            end
        else
            # More aggressive for Float64: geometric steps spanning multiple orders of magnitude.
            r = strength === :strong ? T(1.3) : T(1.2)
            @inbounds for i in 1:n_steps
                steps[i] = r^(i - 1)
            end
            steps ./= sum(steps)
        end

        x = Vector{T}(undef, n)
        x[1] = zero(T)
        @inbounds for i in 1:n_steps
            x[i + 1] = x[i] + steps[i]
        end
        x[end] = one(T)
        @assert all(diff(x) .> zero(T))
        return x
    end

    function _build_tridiagonal_derivative_bc(x::AbstractVector{T}, bc_pair::FI.BCPair{T}) where {T<:AbstractFloat}
        spacing = FI._create_spacing(x)
        n = length(x) - 1
        dl = Vector{T}(undef, n)
        d_diag = Vector{T}(undef, n + 1)
        du = Vector{T}(undef, n)

        FI._set_first_row!(d_diag, du, bc_pair.left, spacing)
        FI._set_last_row!(dl, d_diag, bc_pair.right, spacing)

        @inbounds for i in 2:n
            h_im1 = FI._get_h(spacing, i - 1)
            h_i = FI._get_h(spacing, i)
            dl[i - 1] = h_im1
            d_diag[i] = 2 * (h_im1 + h_i)
            du[i] = h_i
        end

        A = LA.Tridiagonal(dl, d_diag, du)
        return spacing, A
    end

    function _build_tridiagonal_periodic_Aprime(x::AbstractVector{T}) where {T<:AbstractFloat}
        spacing = FI._create_spacing(x)
        n_sys = length(x) - 1
        dl = Vector{T}(undef, n_sys - 1)
        d_diag = Vector{T}(undef, n_sys)
        du = Vector{T}(undef, n_sys - 1)

        h_n = FI._get_h(spacing, n_sys)
        h_1 = FI._get_h(spacing, 1)
        d_diag[1] = h_n + 2 * h_1

        @inbounds for i in 2:(n_sys - 1)
            h_im1 = FI._get_h(spacing, i - 1)
            h_i = FI._get_h(spacing, i)
            dl[i - 1] = h_im1
            d_diag[i] = 2 * (h_im1 + h_i)
            du[i - 1] = h_i
        end

        h_nm1 = FI._get_h(spacing, n_sys - 1)
        dl[n_sys - 1] = h_nm1
        d_diag[n_sys] = 2 * h_nm1 + h_n
        du[n_sys - 1] = h_nm1

        A = LA.Tridiagonal(dl, d_diag, du)
        return spacing, A
    end

    function _compare_solutions(x_custom, x_base; rtol::Real)
        den = max(maximum(abs.(x_base)), eps(eltype(x_base)))
        rel = maximum(abs.(x_custom .- x_base)) / den
        @test rel <= rtol
    end

    @testset "Derivative-BC (build A + lu + ldiv! in-test)" begin
        Random.seed!(0)

        for T in (Float64, Float32)
            for (xname, x) in (
                ("uniform", _x_uniform(T, 101)),
                ("nonuniform", _x_nonuniform(T, 101; seed=1)),
                ("strongly-nonuniform", _x_strongly_nonuniform(T, 101; strength=:strong)),
            )
                for bc_in in (NaturalBC(), ClampedBC(), FI.BCPair(FI.Deriv1(T(0.25)), FI.Deriv2(T(0))))
                    bc = FI._normalize_bc(bc_in, T)
                    spacing, A = _build_tridiagonal_derivative_bc(x, bc)

                    y = rand(T, length(x))
                    rhs = similar(y)
                    FI.compute_rhs!(rhs, y, x, spacing, bc)

                    # Baseline: stdlib LU + ldiv! (default pivot strategy)
                    x_base = copy(rhs)
                    F = LA.lu(A)
                    LA.ldiv!(x_base, F, rhs)

                    # Custom: LU(NoPivot) + cached inv_d + Thomas backward substitution
                    lu_np = LA.lu(A, LA.NoPivot())
                    inv_d = similar(lu_np.factors.d)
                    @inbounds for i in eachindex(inv_d)
                        inv_d[i] = inv(lu_np.factors.d[i])
                    end
                    x_custom = copy(rhs)
                    FI._ldiv_tridiagonal_nopiv!(x_custom, lu_np, inv_d)

                    # Residuals should be comparable
                    resid_base = maximum(abs.(A * x_base .- rhs))
                    resid_custom = maximum(abs.(A * x_custom .- rhs))
                    @test resid_custom <= resid_base * T(10) + eps(T) * T(100)

                    # Relative agreement threshold (Float32 looser, strongly-nonuniform looser)
                    rtol = if T === Float64
                        xname == "strongly-nonuniform" ? 1e-8 : 1e-10
                    else
                        xname == "strongly-nonuniform" ? 1e-4 : 1e-5
                    end
                    _compare_solutions(x_custom, x_base; rtol=rtol)
                end
            end
        end
    end

    @testset "Periodic A' (build A' + lu + ldiv! in-test)" begin
        Random.seed!(1)

        for T in (Float64, Float32)
            for (xname, x) in (
                ("uniform", _x_uniform(T, 101)),
                ("nonuniform", _x_nonuniform(T, 101; seed=2)),
                ("strongly-nonuniform", _x_strongly_nonuniform(T, 101; strength=:strong)),
            )
                spacing, Aprime = _build_tridiagonal_periodic_Aprime(x)

                y = rand(T, length(x))
                rhs = Vector{T}(undef, size(Aprime, 1))
                FI.compute_rhs_periodic!(rhs, y, spacing)

                x_base = copy(rhs)
                F = LA.lu(Aprime)
                LA.ldiv!(x_base, F, rhs)

                lu_np = LA.lu(Aprime, LA.NoPivot())
                inv_d = similar(lu_np.factors.d)
                @inbounds for i in eachindex(inv_d)
                    inv_d[i] = inv(lu_np.factors.d[i])
                end
                x_custom = copy(rhs)
                FI._ldiv_tridiagonal_nopiv!(x_custom, lu_np, inv_d)

                resid_base = maximum(abs.(Aprime * x_base .- rhs))
                resid_custom = maximum(abs.(Aprime * x_custom .- rhs))
                @test resid_custom <= resid_base * T(10) + eps(T) * T(100)

                rtol = if T === Float64
                    xname == "strongly-nonuniform" ? 1e-8 : 1e-10
                else
                    xname == "strongly-nonuniform" ? 1e-4 : 1e-5
                end
                _compare_solutions(x_custom, x_base; rtol=rtol)
            end
        end
    end
end
