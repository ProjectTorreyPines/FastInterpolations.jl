@testitem "Thomas LU Solver" begin
    using Random

    const FI = FastInterpolations
    const LA = FI.LinearAlgebra


    # ─────────────────────────────────────────────────────────────────────────
    # Helper functions for generating test grids
    # ─────────────────────────────────────────────────────────────────────────

    function _x_uniform(::Type{T}, n::Int) where {T <: AbstractFloat}
        return collect(range(T(0), T(1), n))
    end

    function _x_nonuniform(::Type{T}, n::Int; seed::Int = 0) where {T <: AbstractFloat}
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

    function _x_strongly_nonuniform(::Type{T}, n::Int; strength::Symbol = :strong) where {T <: AbstractFloat}
        # Construct strictly-increasing x with a large step ratio while keeping Float32 representable.
        n_steps = n - 1
        steps = Vector{T}(undef, n_steps)
        if T === Float32
            # Safe for Float32: split steps into tiny and large blocks (ratio ~ O(1e2-1e3)).
            k = n_steps ÷ 2
            small = strength === :strong ? T(1.0e-4) : T(1.0e-3)
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

    # ─────────────────────────────────────────────────────────────────────────
    # Helper functions for building tridiagonal matrices
    # ─────────────────────────────────────────────────────────────────────────

    function _build_tridiagonal_derivative_bc(x::AbstractVector{T}, bc_pair::FI.BCPair) where {T <: AbstractFloat}
        cache_x = FI._cache_axis(FI._convert_copy(x, T), FI.NoBC())
        n = length(cache_x) - 1
        dl = Vector{T}(undef, n)
        d_diag = Vector{T}(undef, n + 1)
        du = Vector{T}(undef, n)

        FI._set_first_row!(d_diag, du, bc_pair.left, cache_x)
        FI._set_last_row!(dl, d_diag, bc_pair.right, cache_x)

        @inbounds for i in 2:n
            h_im1 = FI._get_h(cache_x, i - 1)
            h_i = FI._get_h(cache_x, i)
            dl[i - 1] = h_im1
            d_diag[i] = 2 * (h_im1 + h_i)
            du[i] = h_i
        end

        A = LA.Tridiagonal(dl, d_diag, du)
        return cache_x, A
    end

    function _build_tridiagonal_periodic_Aprime(x::AbstractVector{T}) where {T <: AbstractFloat}
        cache_x = FI._cache_axis(FI._convert_copy(x, T), FI.NoBC())
        n_sys = length(cache_x) - 1
        dl = Vector{T}(undef, n_sys - 1)
        d_diag = Vector{T}(undef, n_sys)
        du = Vector{T}(undef, n_sys - 1)

        h_n = FI._get_h(cache_x, n_sys)
        h_1 = FI._get_h(cache_x, 1)
        d_diag[1] = h_n + 2 * h_1

        @inbounds for i in 2:(n_sys - 1)
            h_im1 = FI._get_h(cache_x, i - 1)
            h_i = FI._get_h(cache_x, i)
            dl[i - 1] = h_im1
            d_diag[i] = 2 * (h_im1 + h_i)
            du[i - 1] = h_i
        end

        h_nm1 = FI._get_h(cache_x, n_sys - 1)
        dl[n_sys - 1] = h_nm1
        d_diag[n_sys] = 2 * h_nm1 + h_n
        du[n_sys - 1] = h_nm1

        A = LA.Tridiagonal(dl, d_diag, du)
        return cache_x, A
    end

    # ═════════════════════════════════════════════════════════════════════════
    # Section 1: Core Thomas Algorithm (_ldiv_tridiagonal_nopiv!)
    # ═════════════════════════════════════════════════════════════════════════

    @testset "Derivative-BC: ThomasFactorization vs stdlib LU" begin
        Random.seed!(0)

        for T in (Float64, Float32)
            for (xname, x) in (
                    ("uniform", _x_uniform(T, 101)),
                    ("nonuniform", _x_nonuniform(T, 101; seed = 1)),
                    ("strongly-nonuniform", _x_strongly_nonuniform(T, 101; strength = :strong)),
                )
                for bc_in in (ZeroCurvBC(), ZeroSlopeBC(), FI.BCPair(FI.Deriv1(T(0.25)), FI.Deriv2(T(0))))
                    bc = FI._normalize_bc(bc_in, T)
                    cache_x, A = _build_tridiagonal_derivative_bc(x, bc)

                    y = rand(T, length(x))
                    rhs = similar(y)
                    FI.compute_rhs!(rhs, y, cache_x, bc)

                    # Baseline: stdlib LU + ldiv! (default pivot strategy)
                    x_base = copy(rhs)
                    F = LA.lu(A)
                    LA.ldiv!(x_base, F, rhs)

                    # Custom: ThomasFactorization (copy arrays since factorize! modifies in-place)
                    dl = copy(A.dl)
                    d_diag = copy(A.d)
                    du = copy(A.du)
                    thomas = FI.thomas_factorize!(dl, d_diag, du)
                    x_custom = copy(rhs)
                    FI._ldiv_tridiagonal_nopiv!(x_custom, thomas)

                    # Residuals should be comparable
                    resid_base = maximum(abs.(A * x_base .- rhs))
                    resid_custom = maximum(abs.(A * x_custom .- rhs))
                    @test resid_custom <= resid_base * T(10) + eps(T) * T(100)

                    @test x_custom ≈ x_base rtol = 5 * eps(T)
                end
            end
        end
    end

    @testset "Periodic A': ThomasFactorization vs stdlib LU" begin
        Random.seed!(1)

        for T in (Float64, Float32)
            for (xname, x) in (
                    ("uniform", _x_uniform(T, 101)),
                    ("nonuniform", _x_nonuniform(T, 101; seed = 2)),
                    ("strongly-nonuniform", _x_strongly_nonuniform(T, 101; strength = :strong)),
                )
                cache_x, Aprime = _build_tridiagonal_periodic_Aprime(x)

                y = rand(T, length(x))
                rhs = Vector{T}(undef, size(Aprime, 1))
                # `compute_rhs_periodic!` reads cell widths via `_get_h(cache_x, i)`.
                # Inclusive form: cache_x is the user's length n+1 grid (closed
                # cycle); `_get_h(cache_x, n)` returns the last real cell.
                cache_x = FastInterpolations._cache_axis(FastInterpolations._convert_copy(x, T), FastInterpolations.NoBC())
                FI.compute_rhs_periodic!(rhs, y, cache_x)

                # Baseline: stdlib LU + ldiv!
                x_base = copy(rhs)
                F = LA.lu(Aprime)
                LA.ldiv!(x_base, F, rhs)

                # Custom: ThomasFactorization
                dl = copy(Aprime.dl)
                d_diag = copy(Aprime.d)
                du = copy(Aprime.du)
                thomas = FI.thomas_factorize!(dl, d_diag, du)
                x_custom = copy(rhs)
                FI._ldiv_tridiagonal_nopiv!(x_custom, thomas)

                resid_base = maximum(abs.(Aprime * x_base .- rhs))
                resid_custom = maximum(abs.(Aprime * x_custom .- rhs))
                @test resid_custom <= resid_base * T(10) + eps(T) * T(100)

                @test x_custom ≈ x_base rtol = 5 * eps(T)
            end
        end
    end

    @testset "Periodic q vector (Sherman-Morrison u solve)" begin
        # Tests the specific solve used in _build_periodic_cache for computing
        # q = A'^{-1} * u where u = [1, 0, ..., 0, 1]^T
        Random.seed!(42)

        for T in (Float64, Float32)
            for (xname, x) in (
                    ("uniform", _x_uniform(T, 51)),
                    ("nonuniform", _x_nonuniform(T, 51; seed = 3)),
                    ("strongly-nonuniform", _x_strongly_nonuniform(T, 51; strength = :strong)),
                )
                cache_x, Aprime = _build_tridiagonal_periodic_Aprime(x)
                n_sys = size(Aprime, 1)

                # Baseline: stdlib LU + ldiv!
                u_base = zeros(T, n_sys)
                u_base[1] = one(T)
                u_base[end] = one(T)
                F_base = LA.lu(Aprime)
                q_base = F_base \ u_base

                # Custom: ThomasFactorization
                dl = copy(Aprime.dl)
                d_diag = copy(Aprime.d)
                du = copy(Aprime.du)
                thomas = FI.thomas_factorize!(dl, d_diag, du)
                q_custom = zeros(T, n_sys)
                q_custom[1] = one(T)
                q_custom[end] = one(T)
                FI._ldiv_tridiagonal_nopiv!(q_custom, thomas)

                # Residual check: A' * q ≈ u
                u_ref = zeros(T, n_sys)
                u_ref[1] = one(T)
                u_ref[end] = one(T)
                resid_base = maximum(abs.(Aprime * q_base .- u_ref))
                resid_custom = maximum(abs.(Aprime * q_custom .- u_ref))
                @test resid_custom <= resid_base * T(10) + eps(T) * T(100)

                # Solution agreement
                @test q_custom ≈ q_base rtol = 5 * eps(T)
            end
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # Section 2: Batch Thomas Algorithm (_ldiv_along_dim!)
    # ═════════════════════════════════════════════════════════════════════════

    @testset "Batch _ldiv_along_dim!" begin
        @testset "Val(1) throws ArgumentError" begin
            x = collect(range(0.0, 1.0, 20))
            cache = FI.CubicSplineCache(x; bc = ZeroCurvBC())
            z = rand(20, 5)

            @test_throws ArgumentError FI._ldiv_along_dim!(z, cache.thomas, Val(1))
        end

        @testset "Val(2) correctness: batch == sequential" begin
            for T in (Float64, Float32)
                for n in (10, 50, 101)
                    x = collect(range(T(0), T(1), n))
                    cache = FI.CubicSplineCache(x; bc = ZeroCurvBC())

                    n_batch = 20
                    z_batch = rand(T, n_batch, n)
                    z_seq = copy(z_batch)

                    # Sequential reference: solve each RHS row independently
                    @inbounds for i in 1:n_batch
                        FI._ldiv_tridiagonal_nopiv!(view(z_seq, i, :), cache.thomas)
                    end

                    FI._ldiv_along_dim!(z_batch, cache.thomas, Val(2))

                    @test z_batch ≈ z_seq rtol = eps(T) * 200
                end
            end
        end

        @testset "Periodic system size consistency" begin
            x = collect(range(0.0, 1.0, 51))
            cache = FI.CubicSplineCache(x; bc = PeriodicBC())

            n_sys = length(cache.thomas.inv_d) # = length(x)-1
            z = rand(8, n_sys)
            z_ref = copy(z)

            @inbounds for i in 1:size(z, 1)
                FI._ldiv_tridiagonal_nopiv!(view(z_ref, i, :), cache.thomas)
            end

            FI._ldiv_along_dim!(z, cache.thomas, Val(2))
            @test z ≈ z_ref rtol = 1.0e-10
        end
    end
end

# ========================================
# Witness-typed Thomas core (duck grids)
# ========================================
# The factorization mixes three unit spaces — l (dimensionless), du (X),
# inv_d (1/X) — so the storage is per-field typed and the solve writes its
# result (B/X space) into a caller-provided output instead of overwriting the
# RHS. Real grids must stay bit-identical: the reference implementations below
# are the pre-refactor in-place loops, op-for-op (muladd order preserved), used
# as an executable spec.

@testitem "Thomas duck core: Real bit-identity vs reference algorithm" begin
    const FI = FastInterpolations

    # ── pre-refactor loops (executable spec; do not "simplify") ──
    function ref_factorize!(dl, d, du)
        n = length(d)
        @inbounds begin
            inv_d_val = inv(d[1])
            d[1] = inv_d_val
            for i in 1:(n - 1)
                l_val = dl[i] * inv_d_val
                dl[i] = l_val
                d_next = d[i + 1] - l_val * du[i]
                inv_d_val = inv(d_next)
                d[i + 1] = inv_d_val
            end
        end
        return dl, du, d   # (l, du, inv_d)
    end
    function ref_nopiv!(b, l, du, inv_d)
        n = length(inv_d)
        @inbounds for i in 1:(n - 1)
            b[i + 1] = muladd(-l[i], b[i], b[i + 1])
        end
        @inbounds b[n] *= inv_d[n]
        @inbounds for i in (n - 1):-1:1
            b[i] = muladd(-du[i], b[i + 1], b[i]) * inv_d[i]
        end
        return b
    end
    function ref_transpose!(b, l, du, inv_d)
        n = length(inv_d)
        @inbounds b[1] *= inv_d[1]
        @inbounds for i in 2:n
            b[i] = muladd(-du[i - 1], b[i - 1], b[i]) * inv_d[i]
        end
        @inbounds for i in (n - 1):-1:1
            b[i] = muladd(-l[i], b[i + 1], b[i])
        end
        return b
    end

    mk(n) = (
        d = [4.0 + sin(i) for i in 1:n],
        dl = [1.0 + 0.1 * cos(i) for i in 1:(n - 1)],
        du = [1.0 + 0.1 * sin(2i) for i in 1:(n - 1)],
        b = [sin(3i) for i in 1:n],
    )
    bitsame(a, b) = length(a) == length(b) && all(a[i] === b[i] for i in eachindex(a))

    for n in (5, 64)
        t = mk(n)
        l_r, du_r, inv_r = ref_factorize!(copy(t.dl), copy(t.d), copy(t.du))

        F = FI.thomas_factorize(t.dl, t.d, t.du)
        @test typeof(F.dl) === Vector{Float64}
        @test typeof(F.inv_d) === Vector{Float64}
        @test F.du === t.du                    # stored by reference, unchanged
        @test bitsame(F.dl, l_r)
        @test bitsame(F.inv_d, inv_r)
        # inputs are read-only now
        fresh = mk(n)
        @test bitsame(t.dl, fresh.dl) && bitsame(t.d, fresh.d)

        # nopiv: alias form ≡ old in-place, bit-for-bit
        b_ref = ref_nopiv!(copy(t.b), l_r, du_r, inv_r)
        b_alias = copy(t.b)
        @test FI._ldiv_tridiagonal_nopiv!(b_alias, b_alias, F) === b_alias
        @test bitsame(b_alias, b_ref)
        # nopiv: separate-output form (b is forward-sweep scratch)
        x = similar(t.b)
        FI._ldiv_tridiagonal_nopiv!(x, copy(t.b), F)
        @test bitsame(x, b_ref)

        # transpose: alias + separate-output, bit-for-bit
        bt_ref = ref_transpose!(copy(t.b), l_r, du_r, inv_r)
        bt_alias = copy(t.b)
        @test FI._ldiv_tridiagonal_transpose!(bt_alias, bt_alias, F) === bt_alias
        @test bitsame(bt_alias, bt_ref)
        xt = similar(t.b)
        FI._ldiv_tridiagonal_transpose!(xt, copy(t.b), F)
        @test bitsame(xt, bt_ref)

        # batch Val{2} consumes the new struct; rows ≡ per-row nopiv
        z = [sin(i + j) for i in 1:3, j in 1:n]
        z_ref = copy(z)
        for i in 1:3
            row = z_ref[i, :]
            FI._ldiv_tridiagonal_nopiv!(row, row, F)
            z_ref[i, :] = row
        end
        FI._ldiv_along_dim!(z, F, Val(2))
        @test z ≈ z_ref rtol = 1.0e-12
        @test_throws ArgumentError FI._ldiv_along_dim!(z, F, Val(1))
    end
end

@testitem "Thomas duck core: unit-carrying axis solves in witness spaces" begin
    using Unitful
    const FI = FastInterpolations

    n = 6
    d = [(4.0 + sin(i)) * u"s" for i in 1:n]
    dl = [(1.0 + 0.1 * cos(i)) * u"s" for i in 1:(n - 1)]
    du = [(1.0 + 0.1 * sin(2i)) * u"s" for i in 1:(n - 1)]

    F = FI.thomas_factorize(dl, d, du)
    @test eltype(F.dl) === Float64                    # L multipliers: dimensionless
    @test eltype(F.inv_d) === typeof(inv(1.0u"s"))    # 1/X
    @test F.du === du                                 # X, by reference

    # value-preservation: identical mantissas to the ustrip'd Real solve
    Fr = FI.thomas_factorize(ustrip.(u"s", dl), ustrip.(u"s", d), ustrip.(u"s", du))
    @test all(F.dl[i] === Fr.dl[i] for i in eachindex(Fr.dl))
    @test all(ustrip(u"s^-1", F.inv_d[i]) === Fr.inv_d[i] for i in eachindex(Fr.inv_d))

    # cubic-shaped RHS: Y/X in, Y/X² out
    b = [sin(3i) * u"W/s" for i in 1:n]
    x = Vector{typeof(1.0u"W/s^2")}(undef, n)
    FI._ldiv_tridiagonal_nopiv!(x, copy(b), F)
    xr = Vector{Float64}(undef, n)
    FI._ldiv_tridiagonal_nopiv!(xr, ustrip.(u"W/s", b), Fr)
    @test all(ustrip(u"W/s^2", x[i]) === xr[i] for i in 1:n)

    # transpose (adjoint path): same spaces
    xt = Vector{typeof(1.0u"W/s^2")}(undef, n)
    FI._ldiv_tridiagonal_transpose!(xt, copy(b), F)
    xtr = Vector{Float64}(undef, n)
    FI._ldiv_tridiagonal_transpose!(xtr, ustrip.(u"W/s", b), Fr)
    @test all(ustrip(u"W/s^2", xt[i]) === xtr[i] for i in 1:n)
end

@testitem "Thomas duck core: Dual grid keeps flowing (Dual <: Real)" begin
    using ForwardDiff: Dual, value
    const FI = FastInterpolations

    n = 5
    d = Dual.([4.0 + sin(i) for i in 1:n], 1.0)
    dl = Dual.([1.0 + 0.1 * cos(i) for i in 1:(n - 1)], 0.0)
    du = Dual.([1.0 + 0.1 * sin(2i) for i in 1:(n - 1)], 0.0)

    F = FI.thomas_factorize(dl, d, du)
    @test eltype(F.dl) <: Dual
    @test eltype(F.inv_d) <: Dual

    b = Dual.([sin(3i) for i in 1:n], 1.0)
    x = similar(b)
    FI._ldiv_tridiagonal_nopiv!(x, copy(b), F)
    @test eltype(x) <: Dual
    @test all(isfinite, value.(x))
end
