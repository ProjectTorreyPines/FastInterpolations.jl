#!/usr/bin/env julia

using FastInterpolations
using BenchmarkTools
import LinearAlgebra
using Random

const FI = FastInterpolations

function _ns_per_call(t_ns::Integer, n_calls::Integer)
    return Float64(t_ns) / Float64(n_calls)
end

function _build_tridiagonal_derivative_bc(x::AbstractVector{T}, spacing, bc::FI.BCPair{T}) where {T<:AbstractFloat}
    n = length(x) - 1
    dl = Vector{T}(undef, n)
    d_diag = Vector{T}(undef, n + 1)
    du = Vector{T}(undef, n)

    FI._set_first_row!(d_diag, du, bc.left, spacing)
    FI._set_last_row!(dl, d_diag, bc.right, spacing)

    @inbounds for i in 2:n
        h_im1 = FI._get_h(spacing, i - 1)
        h_i = FI._get_h(spacing, i)
        dl[i - 1] = h_im1
        d_diag[i] = 2 * (h_im1 + h_i)
        du[i] = h_i
    end

    return LinearAlgebra.Tridiagonal(dl, d_diag, du)
end

function _build_tridiagonal_periodic_Aprime(x::AbstractVector{T}, spacing) where {T<:AbstractFloat}
    n = length(x) - 1
    dl = Vector{T}(undef, n - 1)
    d_diag = Vector{T}(undef, n)
    du = Vector{T}(undef, n - 1)

    h_n = FI._get_h(spacing, n)
    h_1 = FI._get_h(spacing, 1)
    d_diag[1] = h_n + 2 * h_1

    @inbounds for i in 2:n-1
        h_im1 = FI._get_h(spacing, i - 1)
        h_i = FI._get_h(spacing, i)
        dl[i - 1] = h_im1
        d_diag[i] = 2 * (h_im1 + h_i)
        du[i - 1] = h_i
    end

    h_nm1 = FI._get_h(spacing, n - 1)
    dl[n - 1] = h_nm1
    d_diag[n] = 2 * h_nm1 + h_n
    du[n - 1] = h_nm1

    return LinearAlgebra.Tridiagonal(dl, d_diag, du)
end

function _time_loop!(f!, n_warmup::Int, n_iter::Int)
    for _ in 1:n_warmup
        f!()
    end
    GC.gc()
    prev_gc = GC.enable(false)
    try
        t0 = time_ns()
        for _ in 1:n_iter
            f!()
        end
        t1 = time_ns()
        return t1 - t0
    finally
        GC.enable(prev_gc)
    end
end

function bench_derivative_bc(; T::Type{<:AbstractFloat}, n::Int, seconds::Float64)
    x = collect(range(T(0), T(1), n))
    bc = FI._normalize_bc(NaturalBC(), T)
    cache = FI.CubicSplineCache(x; bc=bc)
    A = _build_tridiagonal_derivative_bc(cache.x, cache.spacing, bc)

    xv = cache.x
    spacing = cache.spacing
    lu = cache.lu_factor
    inv_d = cache.inv_d

    y = rand(T, n)
    d = similar(y)
    out = similar(y)

    baseline!() = begin
        FI.compute_rhs!(d, y, xv, spacing, bc)
        LinearAlgebra.ldiv!(out, lu, d)
        return nothing
    end

    custom!() = begin
        FI.compute_rhs!(out, y, xv, spacing, bc)
        FI._ldiv_tridiagonal_nopiv!(out, lu, inv_d)
        return nothing
    end

    baseline!() # out = baseline solution, d = rhs
    out_base = copy(out)
    rhs = copy(d)
    resid_base = maximum(abs.(A * out_base .- rhs))

    custom!() # out = custom solution
    out_cust = copy(out)
    resid_cust = maximum(abs.(A * out_cust .- rhs))

    sol_diff = maximum(abs.(out_cust .- out_base))

    t_base = @benchmark begin
        FI.compute_rhs!($d, $y, $xv, $spacing, $bc)
        LinearAlgebra.ldiv!($out, $lu, $d)
    end evals = 1 seconds = seconds

    t_cust = @benchmark begin
        FI.compute_rhs!($out, $y, $xv, $spacing, $bc)
        FI._ldiv_tridiagonal_nopiv!($out, $lu, $inv_d)
    end evals = 1 seconds = seconds

    est_base = median(t_base)
    est_cust = median(t_cust)
    return (
        sol_diff=sol_diff,
        resid_base=resid_base,
        resid_cust=resid_cust,
        est_base=est_base,
        est_cust=est_cust,
    )
end

function bench_periodic_tridiagonal(; T::Type{<:AbstractFloat}, n::Int, seconds::Float64)
    # Periodic uses n points, but the cyclic system is size n-1 (A') internally.
    x = collect(range(T(0), T(1), n))
    cache = FI.CubicSplineCache(x; bc=PeriodicBC())
    Aprime = _build_tridiagonal_periodic_Aprime(cache.x, cache.spacing)

    spacing = cache.spacing
    lu = cache.lu_factor
    inv_d = cache.inv_d

    y = rand(T, n)
    n_sys = length(cache.inv_d) # = n-1
    rhs = rand(T, n_sys)
    tmp = similar(rhs)

    baseline!() = begin
        FI.compute_rhs_periodic!(rhs, y, spacing)
        LinearAlgebra.ldiv!(tmp, lu, rhs)
        return nothing
    end

    custom!() = begin
        FI.compute_rhs_periodic!(tmp, y, spacing)
        FI._ldiv_tridiagonal_nopiv!(tmp, lu, inv_d)
        return nothing
    end

    baseline!()
    tmp_base = copy(tmp)
    rhs_snap = copy(rhs)
    resid_base = maximum(abs.(Aprime * tmp_base .- rhs_snap))

    custom!()
    tmp_cust = copy(tmp)
    resid_cust = maximum(abs.(Aprime * tmp_cust .- rhs_snap))

    sol_diff = maximum(abs.(tmp_cust .- tmp_base))

    t_base = @benchmark begin
        FI.compute_rhs_periodic!($rhs, $y, $spacing)
        LinearAlgebra.ldiv!($tmp, $lu, $rhs)
    end evals = 1 seconds = seconds

    t_cust = @benchmark begin
        FI.compute_rhs_periodic!($tmp, $y, $spacing)
        FI._ldiv_tridiagonal_nopiv!($tmp, $lu, $inv_d)
    end evals = 1 seconds = seconds

    est_base = median(t_base)
    est_cust = median(t_cust)
    return (
        sol_diff=sol_diff,
        resid_base=resid_base,
        resid_cust=resid_cust,
        est_base=est_base,
        est_cust=est_cust,
    )
end

function main()
    Random.seed!(0)

    seconds = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 2.0

    println("FastInterpolations tridiagonal solve benchmark")
    println("  seconds per benchmark = $seconds")
    println()
    println("Case, T, n, sol_diff_inf, resid_inf baseline, resid_inf custom, median ns baseline, median ns custom, speedup, allocs baseline, allocs custom, bytes baseline, bytes custom")

    for T in (Float64, Float32)
        for n in (51, 201, 1001)
            r = bench_derivative_bc(; T=T, n=n, seconds=seconds)
            t_base = Float64(r.est_base.time)
            t_cust = Float64(r.est_cust.time)
            speedup = t_base / t_cust
            println(
                "DerivativeBC, $T, $n, $(r.sol_diff), $(r.resid_base), $(r.resid_cust), " *
                "$(t_base), $(t_cust), $(speedup), " *
                "$(r.est_base.allocs), $(r.est_cust.allocs), $(r.est_base.memory), $(r.est_cust.memory)"
            )
        end
    end

    for T in (Float64, Float32)
        for n in (51, 201, 1001)
            r = bench_periodic_tridiagonal(; T=T, n=n, seconds=seconds)
            t_base = Float64(r.est_base.time)
            t_cust = Float64(r.est_cust.time)
            speedup = t_base / t_cust
            println(
                "Periodic(A'), $T, $n, $(r.sol_diff), $(r.resid_base), $(r.resid_cust), " *
                "$(t_base), $(t_cust), $(speedup), " *
                "$(r.est_base.allocs), $(r.est_cust.allocs), $(r.est_base.memory), $(r.est_cust.memory)"
            )
        end
    end
end

main()
