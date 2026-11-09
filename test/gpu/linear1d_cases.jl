# Shared case builder for the GPU `LinearInterpolant` tests: included by the JLArrays
# extension test (test/ext/test_gpu_linear1d.jl, CI) and the real-GPU test
# (test/gpu/linear1d.jl). Deterministic data, no Random dependency.

gpu_fy(x) = sin(2 * oftype(x, π) * x) + x

# Uniform (→ `_CachedRange`) and non-uniform (→ `_CachedVector`) grids on [0, 1].
gpu_grids(::Type{T}, n::Int) where {T} = (
    ("Range", range(zero(T), one(T); length = n)),
    ("Vector", T[(T(i - 1) / T(n - 1))^T(1.5) for i in 1:n]),
)

# Golden-ratio sequence: in-domain [0, 1) and an OOB mix on [-0.2, 1.2).
function gpu_queries(::Type{T}, nq::Int) where {T}
    q_in = T[mod(T(0.6180339887498949) * k, one(T)) for k in 1:nq]
    q_oob = T(1.4) .* q_in .- T(0.2)
    return q_in, q_oob
end

# Range inputs users pass in. Each becomes an isbits `_CachedRange` with value-typed fields
# (Base `range(0f0, 1f0; length)` keeps Float64 internals, which Metal cannot compile);
# an Int `1:n` takes the unit-step search path.
gpu_range_grids(::Type{T}, n::Int) where {T} = (
    ("1:n", 1:n),
    ("range", range(zero(T), one(T); length = n)),
    ("LinRange", LinRange(zero(T), one(T), n)),
)

# Values and (in-domain, OOB-mix) queries on an arbitrary grid's own domain.
gpu_values(::Type{T}, x) where {T} = gpu_fy.(T.((x .- first(x)) ./ (last(x) - first(x))))
function gpu_queries_on(::Type{T}, x, nq::Int) where {T}
    q_in, q_oob = gpu_queries(T, nq)
    lo, w = T(first(x)), T(last(x) - first(x))
    return lo .+ w .* q_in, lo .+ w .* q_oob
end

# Non-throwing extraps evaluated under broadcast (NoExtrap is batch-only on GPUs).
gpu_extraps(::Type{T}) where {T} = (
    ("InBounds", InBounds(), :in),
    ("Clamp", ClampExtrap(), :oob),
    ("Fill", FillExtrap(T(NaN)), :oob),
    ("Extend", ExtendExtrap(), :oob),
)

# Elementwise closeness scaled by the data magnitude: a pure rtol is meaningless near
# zeros of y, and backend FP codegen may differ from the CPU by a few ulps.
function gpu_close(dev::AbstractArray, cpu::AbstractArray, scale)
    d = Array(dev)
    size(d) == size(cpu) || return false
    tol = 4 * eps(eltype(cpu))
    return all(zip(d, cpu)) do (a, b)
        (isnan(a) && isnan(b)) || abs(a - b) <= tol * max(abs(b), scale)
    end
end
