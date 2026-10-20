# Shared case builder for the GPU `LinearInterpolantND` tests: included by the JLArrays
# extension test (test/ext/test_gpu_linear_nd.jl, CI) and the local Metal test
# (test/gpu/metal_linear_nd.jl). Deterministic data, no Random dependency.

include(joinpath(@__DIR__, "linear1d_cases.jl"))     # gpu_close, gpu_fy

# 2D and 3D grids mixing uniform (→ `_CachedRange`) and non-uniform (→ device vector) axes.
nonuniform(::Type{T}, n) where {T} = T[(T(i - 1) / T(n - 1))^T(1.5) for i in 1:n]
nd_grid_cases(::Type{T}) where {T} = (
    ("2D range×range", (range(zero(T), one(T); length = 33), range(zero(T), one(T); length = 17))),
    ("2D range×vector", (range(zero(T), one(T); length = 33), nonuniform(T, 17))),
    ("3D vector×range×vector", (nonuniform(T, 13), range(zero(T), one(T); length = 9), nonuniform(T, 11))),
)

nd_values(::Type{T}, grids) where {T} = [T(sum(gpu_fy, p)) for p in Iterators.product(grids...)]

# Golden-ratio queries per axis (decorrelated by a per-axis offset): in-domain [0, 1) and an
# OOB mix on [-0.2, 1.2); returned as a tuple of vectors (SoA).
function nd_queries(::Type{T}, N::Int, nq::Int) where {T}
    q_in = ntuple(d -> T[mod(T(0.6180339887498949) * k + T(0.137) * d, one(T)) for k in 1:nq], N)
    q_oob = map(q -> T(1.4) .* q .- T(0.2), q_in)
    return q_in, q_oob
end

# The same queries as a vector of points (AoS).
nd_points(q::Tuple) = [ntuple(d -> q[d][k], length(q)) for k in eachindex(q[1])]

nd_extraps(::Type{T}) where {T} = (
    ("InBounds", InBounds(), :in),
    ("Clamp", ClampExtrap(), :oob),
    ("Fill", FillExtrap(T(NaN)), :oob),
    ("Extend", ExtendExtrap(), :oob),
)
