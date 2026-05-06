#!/usr/bin/env julia
# Profile PromolecularRef allocation sources
using Pkg; Pkg.activate(".")
using FastInterpolations

# Build a 1D cubic spline (simulates atomic density spline)
r = range(0.0, 10.0, length=200)
rho_data = exp.(-r)  # exponential decay
itp_1d = cubic_interp(r, rho_data; extrap=FillExtrap(0.0))

# ── 1. Dict{Int, Any} boxing ──
# Simulates _wfc_cache with Any values
dict_any = Dict{Int, Any}(1 => itp_1d)
dict_typed = Dict{Int, typeof(itp_1d)}(1 => itp_1d)

# Warmup
dict_any[1](5.0)
dict_typed[1](5.0)

println("=== Dict boxing (lookup + scalar eval) ===")
println("  Dict{Int, Any}:    $(@allocated dict_any[1](5.0)) bytes")
println("  Dict{Int, typed}:  $(@allocated dict_typed[1](5.0)) bytes")

# With deriv
dict_any[1](5.0; deriv=DerivOp{1}())
dict_typed[1](5.0; deriv=DerivOp{1}())

println("\n=== Dict boxing (lookup + deriv eval) ===")
println("  Dict{Int, Any}:    $(@allocated dict_any[1](5.0; deriv=DerivOp{1}())) bytes")
println("  Dict{Int, typed}:  $(@allocated dict_typed[1](5.0; deriv=DerivOp{1}())) bytes")

# ── 2. Direct 1D eval (no Dict) ──
itp_1d(5.0)
itp_1d(5.0; deriv=DerivOp{1}())
itp_1d(5.0; deriv=DerivOp{2}())

println("\n=== Direct 1D cubic eval (no Dict) ===")
println("  value:  $(@allocated itp_1d(5.0)) bytes")
println("  deriv1: $(@allocated itp_1d(5.0; deriv=DerivOp{1}())) bytes")
println("  deriv2: $(@allocated itp_1d(5.0; deriv=DerivOp{2}())) bytes")

# ── 3. Simulate PromolecularRef loop with Any-typed cache ──
# 26 atoms, each with a 1D spline lookup + eval
function sim_promolecular_any!(cache, atoms, q)
    val = 0.0
    for (Z, R) in atoms
        xx1 = q[1] - R[1]; xx2 = q[2] - R[2]; xx3 = q[3] - R[3]
        r = sqrt(xx1^2 + xx2^2 + xx3^2)
        r < 1e-14 && continue
        val += max(cache[Z](r), 0.0)
    end
    return val
end

# Same but with typed cache
function sim_promolecular_typed!(cache, atoms, q)
    val = 0.0
    for (Z, R) in atoms
        xx1 = q[1] - R[1]; xx2 = q[2] - R[2]; xx3 = q[3] - R[3]
        r = sqrt(xx1^2 + xx2^2 + xx3^2)
        r < 1e-14 && continue
        val += max(cache[Z](r), 0.0)
    end
    return val
end

# Same but with direct reference (no Dict)
function sim_promolecular_direct!(itp, atoms, q)
    val = 0.0
    for (Z, R) in atoms
        xx1 = q[1] - R[1]; xx2 = q[2] - R[2]; xx3 = q[3] - R[3]
        r = sqrt(xx1^2 + xx2^2 + xx3^2)
        r < 1e-14 && continue
        val += max(itp(r), 0.0)
    end
    return val
end

# Build fake 26-atom system (all same element)
atoms = [(1, (Float64(i), 0.0, 0.0)) for i in 1:26]
q = (5.0, 0.0, 0.0)

cache_any = Dict{Int, Any}(1 => itp_1d)
cache_typed = Dict{Int, typeof(itp_1d)}(1 => itp_1d)

sim_promolecular_any!(cache_any, atoms, q)
sim_promolecular_typed!(cache_typed, atoms, q)
sim_promolecular_direct!(itp_1d, atoms, q)

println("\n=== Simulated 26-atom PromolecularRef (value only) ===")
println("  Any-typed Dict:   $(@allocated sim_promolecular_any!(cache_any, atoms, q)) bytes")
println("  Typed Dict:       $(@allocated sim_promolecular_typed!(cache_typed, atoms, q)) bytes")
println("  Direct (no Dict): $(@allocated sim_promolecular_direct!(itp_1d, atoms, q)) bytes")

# ── 4. Gradient version ──
function sim_promolecular_grad_any!(cache, atoms, q, ax)
    fp = 0.0
    D1 = DerivOp{1}()
    for (Z, R) in atoms
        xx = (q[1] - R[1], q[2] - R[2], q[3] - R[3])
        r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
        r < 1e-14 && continue
        rhop = cache[Z](r; deriv=D1)
        fp += rhop * xx[ax] / r
    end
    return fp
end

function sim_promolecular_grad_typed!(cache, atoms, q, ax)
    fp = 0.0
    D1 = DerivOp{1}()
    for (Z, R) in atoms
        xx = (q[1] - R[1], q[2] - R[2], q[3] - R[3])
        r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
        r < 1e-14 && continue
        rhop = cache[Z](r; deriv=D1)
        fp += rhop * xx[ax] / r
    end
    return fp
end

function sim_promolecular_grad_direct!(itp, atoms, q, ax)
    fp = 0.0
    D1 = DerivOp{1}()
    for (Z, R) in atoms
        xx = (q[1] - R[1], q[2] - R[2], q[3] - R[3])
        r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
        r < 1e-14 && continue
        rhop = itp(r; deriv=D1)
        fp += rhop * xx[ax] / r
    end
    return fp
end

sim_promolecular_grad_any!(cache_any, atoms, q, 1)
sim_promolecular_grad_typed!(cache_typed, atoms, q, 1)
sim_promolecular_grad_direct!(itp_1d, atoms, q, 1)

println("\n=== Simulated 26-atom PromolecularRef (gradient) ===")
println("  Any-typed Dict:   $(@allocated sim_promolecular_grad_any!(cache_any, atoms, q, 1)) bytes")
println("  Typed Dict:       $(@allocated sim_promolecular_grad_typed!(cache_typed, atoms, q, 1)) bytes")
println("  Direct (no Dict): $(@allocated sim_promolecular_grad_direct!(itp_1d, atoms, q, 1)) bytes")

# ── 5. Hessian axis-finding: comprehension (allocating) vs loop (zero-alloc) ──
# The real PromolecularRef total==2 branch uses:
#   nonzero = [d for d in 1:3 if deriv_order(deriv[d]) > 0]
# which always allocates a small Vector.  Compare with an index-scan loop.

function find_axes_alloc(deriv)
    nonzero = [d for d in 1:3 if deriv_order(deriv[d]) > 0]
    ax1 = nonzero[1]
    ax2 = length(nonzero) >= 2 ? nonzero[2] : ax1
    return ax1, ax2
end

function find_axes_noalloc(deriv)
    ax1 = 0; ax2 = 0
    for d in 1:3
        if deriv_order(deriv[d]) > 0
            if ax1 == 0
                ax1 = d
            else
                ax2 = d
                break
            end
        end
    end
    return ax1, ax2 == 0 ? ax1 : ax2
end

# Representative 3-tuples used when PromolecularRef is called from _phs_eval_with_transform
const deriv_grad    = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}())   # ∂/∂x
const deriv_diag    = (DerivOp{2}(), DerivOp{0}(), DerivOp{0}())   # ∂²/∂x²
const deriv_offdiag = (DerivOp{1}(), DerivOp{1}(), DerivOp{0}())   # ∂²/∂x∂y

find_axes_alloc(deriv_diag);    find_axes_noalloc(deriv_diag)
find_axes_alloc(deriv_offdiag); find_axes_noalloc(deriv_offdiag)

println("\n=== Hessian axis-finding (total==2 branch) ===")
println("  diagonal  ∂²/∂x²   — comprehension: $(@allocated find_axes_alloc(deriv_diag)) bytes   loop: $(@allocated find_axes_noalloc(deriv_diag)) bytes")
println("  off-diag  ∂²/∂x∂y  — comprehension: $(@allocated find_axes_alloc(deriv_offdiag)) bytes   loop: $(@allocated find_axes_noalloc(deriv_offdiag)) bytes")

# ── 6. Full PromolecularRef struct: Any-cache vs typed-cache, all branches ──
# PromolecularRefAny mirrors the current script (cache::Dict{Int,Any}, comprehension in total==2).
# PromolecularRefTyped embeds a typed Dict and uses the non-allocating axis-scan loop.

struct PromolecularRefAny
    atoms::Vector{Tuple{Int, NTuple{3,Float64}}}
    cache::Dict{Int, Any}
end

struct PromolecularRefTyped{I}
    atoms::Vector{Tuple{Int, NTuple{3,Float64}}}
    cache::Dict{Int, I}
end

function (pmr::PromolecularRefAny)(q; deriv=nothing)
    total = deriv === nothing ? 0 : sum(deriv_order(op) for op in deriv)
    if total == 0
        f = 0.0
        for (Z, R) in pmr.atoms
            xx1 = q[1]-R[1]; xx2 = q[2]-R[2]; xx3 = q[3]-R[3]
            r = sqrt(xx1^2 + xx2^2 + xx3^2)
            r < 1e-14 && continue
            f += max(pmr.cache[Z](r), 0.0)
        end
        return f
    end
    if total == 1
        ax = findfirst(d -> deriv_order(deriv[d]) == 1, 1:3)::Int
        fp = 0.0
        D1 = DerivOp{1}()
        for (Z, R) in pmr.atoms
            xx = (q[1]-R[1], q[2]-R[2], q[3]-R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            fp += pmr.cache[Z](r; deriv=D1) * xx[ax] / r
        end
        return fp
    end
    if total == 2
        nonzero = [d for d in 1:3 if deriv_order(deriv[d]) > 0]   # allocating comprehension
        ax1 = nonzero[1]; ax2 = length(nonzero) >= 2 ? nonzero[2] : ax1
        fpp = 0.0
        D1 = DerivOp{1}(); D2 = DerivOp{2}()
        for (Z, R) in pmr.atoms
            xx = (q[1]-R[1], q[2]-R[2], q[3]-R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            rho_itp = pmr.cache[Z]
            rhop  = rho_itp(r; deriv=D1)
            rhopp = rho_itp(r; deriv=D2)
            rfac  = (rhopp - rhop/r) / r^2
            fpp  += ax1 == ax2 ? rhop/r + rfac*xx[ax1]^2 : rfac*xx[ax1]*xx[ax2]
        end
        return fpp
    end
    return 0.0
end

function (pmr::PromolecularRefTyped)(q; deriv=nothing)
    total = deriv === nothing ? 0 : sum(deriv_order(op) for op in deriv)
    if total == 0
        f = 0.0
        for (Z, R) in pmr.atoms
            xx1 = q[1]-R[1]; xx2 = q[2]-R[2]; xx3 = q[3]-R[3]
            r = sqrt(xx1^2 + xx2^2 + xx3^2)
            r < 1e-14 && continue
            f += max(pmr.cache[Z](r), 0.0)
        end
        return f
    end
    if total == 1
        ax = findfirst(d -> deriv_order(deriv[d]) == 1, 1:3)::Int
        fp = 0.0
        D1 = DerivOp{1}()
        for (Z, R) in pmr.atoms
            xx = (q[1]-R[1], q[2]-R[2], q[3]-R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            fp += pmr.cache[Z](r; deriv=D1) * xx[ax] / r
        end
        return fp
    end
    if total == 2
        # Non-allocating axis scan (replaces comprehension)
        ax1 = 0; ax2 = 0
        for d in 1:3
            if deriv_order(deriv[d]) > 0
                if ax1 == 0; ax1 = d
                else ax2 = d; break
                end
            end
        end
        ax2 = ax2 == 0 ? ax1 : ax2
        fpp = 0.0
        D1 = DerivOp{1}(); D2 = DerivOp{2}()
        for (Z, R) in pmr.atoms
            xx = (q[1]-R[1], q[2]-R[2], q[3]-R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            itp = pmr.cache[Z]
            rhop  = itp(r; deriv=D1)
            rhopp = itp(r; deriv=D2)
            rfac  = (rhopp - rhop/r) / r^2
            fpp  += ax1 == ax2 ? rhop/r + rfac*xx[ax1]^2 : rfac*xx[ax1]*xx[ax2]
        end
        return fpp
    end
    return 0.0
end

pmr_any   = PromolecularRefAny(atoms, Dict{Int,Any}(1 => itp_1d))
pmr_typed = PromolecularRefTyped(atoms, Dict{Int,typeof(itp_1d)}(1 => itp_1d))
qpt = (5.0, 0.0, 0.0)

# Warmup all branches
pmr_any(qpt); pmr_typed(qpt)
pmr_any(qpt; deriv=deriv_grad);    pmr_typed(qpt; deriv=deriv_grad)
pmr_any(qpt; deriv=deriv_diag);    pmr_typed(qpt; deriv=deriv_diag)
pmr_any(qpt; deriv=deriv_offdiag); pmr_typed(qpt; deriv=deriv_offdiag)

println("\n=== Full PromolecularRef struct: Any-cache vs Typed-cache (26 atoms) ===")
println("  value (total==0):")
println("    Any-cache:   $(@allocated pmr_any(qpt)) bytes")
println("    Typed-cache: $(@allocated pmr_typed(qpt)) bytes")
println("  gradient ∂/∂x (total==1):")
println("    Any-cache:   $(@allocated pmr_any(qpt; deriv=deriv_grad)) bytes")
println("    Typed-cache: $(@allocated pmr_typed(qpt; deriv=deriv_grad)) bytes")
println("  Hessian ∂²/∂x² diagonal (total==2):")
println("    Any-cache:   $(@allocated pmr_any(qpt; deriv=deriv_diag)) bytes")
println("    Typed-cache: $(@allocated pmr_typed(qpt; deriv=deriv_diag)) bytes")
println("  Hessian ∂²/∂x∂y off-diagonal (total==2):")
println("    Any-cache:   $(@allocated pmr_any(qpt; deriv=deriv_offdiag)) bytes")
println("    Typed-cache: $(@allocated pmr_typed(qpt; deriv=deriv_offdiag)) bytes")

# ── 7. Isolate residual 16-byte allocations in typed-cache gradient/Hessian ──
# Candidates: (a) sum(generator over tuple) for total computation
#             (b) findfirst(closure, 1:3)   for gradient axis lookup

# (a) sum generator over NTuple of DerivOps
function sum_deriv_order_gen(deriv)
    return sum(deriv_order(op) for op in deriv)
end
function sum_deriv_order_loop(deriv)
    s = 0
    for op in deriv
        s += deriv_order(op)
    end
    return s
end

sum_deriv_order_gen(deriv_grad);  sum_deriv_order_loop(deriv_grad)

println("\n=== Residual-alloc isolation: sum(generator) vs plain loop ===")
println("  sum(gen) over grad tuple:  $(@allocated sum_deriv_order_gen(deriv_grad)) bytes")
println("  loop    over grad tuple:   $(@allocated sum_deriv_order_loop(deriv_grad)) bytes")

# (b) findfirst(closure, 1:3) for gradient axis lookup
function ax_findfirst(deriv)
    return findfirst(d -> deriv_order(deriv[d]) == 1, 1:3)::Int
end
function ax_loop(deriv)
    for d in 1:3
        deriv_order(deriv[d]) == 1 && return d
    end
    return 1
end

ax_findfirst(deriv_grad); ax_loop(deriv_grad)

println("\n=== Residual-alloc isolation: findfirst(closure) vs plain loop ===")
println("  findfirst(closure):  $(@allocated ax_findfirst(deriv_grad)) bytes")
println("  loop (no closure):   $(@allocated ax_loop(deriv_grad)) bytes")

# ── 8. PromolecularRefTyped with all overhead eliminated ──
# Replace both findfirst and sum(generator) with plain loops.

function (pmr::PromolecularRefTyped)(q, ::Val{:noalloc}; deriv=nothing)
    total = 0
    if deriv !== nothing
        for op in deriv; total += deriv_order(op); end
    end
    if total == 0
        f = 0.0
        for (Z, R) in pmr.atoms
            xx1 = q[1]-R[1]; xx2 = q[2]-R[2]; xx3 = q[3]-R[3]
            r = sqrt(xx1^2 + xx2^2 + xx3^2)
            r < 1e-14 && continue
            f += max(pmr.cache[Z](r), 0.0)
        end
        return f
    end
    if total == 1
        ax = 0
        for d in 1:3; deriv_order(deriv[d]) == 1 && (ax = d; break); end
        fp = 0.0
        D1 = DerivOp{1}()
        for (Z, R) in pmr.atoms
            xx = (q[1]-R[1], q[2]-R[2], q[3]-R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            fp += pmr.cache[Z](r; deriv=D1) * xx[ax] / r
        end
        return fp
    end
    if total == 2
        ax1 = 0; ax2 = 0
        for d in 1:3
            if deriv_order(deriv[d]) > 0
                if ax1 == 0; ax1 = d
                else ax2 = d; break
                end
            end
        end
        ax2 = ax2 == 0 ? ax1 : ax2
        fpp = 0.0
        D1 = DerivOp{1}(); D2 = DerivOp{2}()
        for (Z, R) in pmr.atoms
            xx = (q[1]-R[1], q[2]-R[2], q[3]-R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            itp = pmr.cache[Z]
            rhop  = itp(r; deriv=D1)
            rhopp = itp(r; deriv=D2)
            rfac  = (rhopp - rhop/r) / r^2
            fpp  += ax1 == ax2 ? rhop/r + rfac*xx[ax1]^2 : rfac*xx[ax1]*xx[ax2]
        end
        return fpp
    end
    return 0.0
end

# Warmup
pmr_typed(qpt, Val(:noalloc))
pmr_typed(qpt, Val(:noalloc); deriv=deriv_grad)
pmr_typed(qpt, Val(:noalloc); deriv=deriv_diag)
pmr_typed(qpt, Val(:noalloc); deriv=deriv_offdiag)

println("\n=== Typed-cache with all closures/generators replaced by loops ===")
println("  value (total==0):               $(@allocated pmr_typed(qpt, Val(:noalloc))) bytes")
println("  gradient ∂/∂x (total==1):       $(@allocated pmr_typed(qpt, Val(:noalloc); deriv=deriv_grad)) bytes")
println("  Hessian diagonal (total==2):    $(@allocated pmr_typed(qpt, Val(:noalloc); deriv=deriv_diag)) bytes")
println("  Hessian off-diag (total==2):    $(@allocated pmr_typed(qpt, Val(:noalloc); deriv=deriv_offdiag)) bytes")

# ── 9. Function-scope validation: rule out global-scope boxing ──
# @allocated calls at global scope can box return values even for type-stable
# functions (the outer interpolated-string context has type Any).
# Wrap in a function to get a true zero-alloc measurement.

function measure_pmr_typed(pmr, qpt, d_grad, d_diag, d_offdiag)
    a0  = @allocated pmr(qpt)
    a1  = @allocated pmr(qpt; deriv=d_grad)
    a2d = @allocated pmr(qpt; deriv=d_diag)
    a2x = @allocated pmr(qpt; deriv=d_offdiag)
    return a0, a1, a2d, a2x
end

# Warmup
measure_pmr_typed(pmr_typed, qpt, deriv_grad, deriv_diag, deriv_offdiag)
a0, a1, a2d, a2x = measure_pmr_typed(pmr_typed, qpt, deriv_grad, deriv_diag, deriv_offdiag)

println("\n=== Typed-cache inside function (eliminates global-scope boxing) ===")
println("  value (total==0):               $a0 bytes")
println("  gradient ∂/∂x (total==1):       $a1 bytes")
println("  Hessian diagonal (total==2):    $a2d bytes")
println("  Hessian off-diag (total==2):    $a2x bytes")
