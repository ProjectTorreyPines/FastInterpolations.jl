# CPU vs Metal performance of LinearInterpolantND (Float32; informational, not a public
# claim). Sections (all by default, or pick by name):
#   nd     : FI CPU batch vs Metal SoA / AoS batch calls and broadcast
#   vs_itp : FastInterpolations vs Interpolations.jl, same grids / data / queries, CPU and Metal
#
#   julia --project=scripts/gpu scripts/gpu/bench_linear_nd.jl [nd] [vs_itp]
#
# Queries: seeded random, OOB mix (10 % beyond each side) with clamp extrapolation (FI
# `ClampExtrap()` ≡ Interpolations `Flat()`). SoA = tuple of coordinate vectors, AoS = vector
# of N-tuples. CPU = single-thread in-place batch. Times µs, minimum.

using FastInterpolations, Adapt, GPUArrays, Metal
import Interpolations
const Itp = Interpolations
include(joinpath(@__DIR__, "metal_timing.jl"))

const T = Float32
const NQS = (10^4, 10^6, 10^7)
GPUArrays.allowscalar(false)

nonuniform(n) = T[(T(i - 1) / T(n - 1))^T(1.5) for i in 1:n]
uniform(n) = range(zero(T), one(T); length = n)
nd_cases() = (
    ("2D range×range 513²", (uniform(513), uniform(513))),
    ("2D range×vector 513²", (uniform(513), nonuniform(513))),
    ("3D range³ 129³", (uniform(129), uniform(129), uniform(129))),
    ("3D vector³ 65³", (nonuniform(65), nonuniform(65), nonuniform(65))),
)
field(grids) = [T(sum(sin(3 * k * v) for (k, v) in enumerate(p))) for p in Iterators.product(grids...)]
soa_queries(N, nq) = ntuple(d -> T(1.2) .* rand(Xoshiro(10 + d), T, nq) .- T(0.1), N)
points(q) = [ntuple(d -> q[d][k], length(q)) for k in eachindex(q[1])]

function nd()
    println("FI LinearInterpolantND — CPU batch vs Metal (ClampExtrap)")
    @printf(
        "%-21s %9s │ %10s │ %8s %8s │ %8s %8s │ %9s │ %8s %9s\n",
        "case", "nq", "CPU batch", "SoA kern", "SoA wall", "AoS kern", "AoS wall", "bcast wal", "CPU/SoA", "|GPU-CPU|"
    )
    for (cname, grids) in nd_cases(), nq in NQS
        N = length(grids)
        itp = linear_interp(grids, field(grids); extrap = ClampExtrap())
        ditp = adapt(MtlArray, itp)
        q = soa_queries(N, nq)
        out = Vector{T}(undef, nq)
        qd, pd, outd = map(MtlArray, q), MtlArray(points(q)), MtlArray(out)
        t_cpu = @belapsed $itp($out, $q)
        k_soa, w_soa = device_time(() -> ditp(outd, qd)), gpu_wall(() -> ditp(outd, qd))
        k_aos, w_aos = device_time(() -> ditp(outd, pd)), gpu_wall(() -> ditp(outd, pd))
        w_bc = gpu_wall(() -> (outd .= ditp.(pd)))
        err = maximum(abs, Array(ditp(qd)) .- itp(q))
        @printf(
            "%-21s %9d │ %10.0f │ %8.0f %8.0f │ %8.0f %8.0f │ %9.0f │ %7.1fx %9.1e\n",
            cname, nq, us(t_cpu), us(k_soa), us(w_soa), us(k_aos), us(w_aos), us(w_bc), t_cpu / w_soa, err
        )
    end
    return
end

# Interpolations gets `LinRange` axes for uniform grids (a Base `range` keeps Float64
# internals that Metal cannot compile) and a gridded interpolant for non-uniform ones.
itp_axis(g) = g isa AbstractRange ? LinRange(first(g), last(g), length(g)) : g

function vs_itp()
    println("FastInterpolations (FI) vs Interpolations.jl (Itp) — ND linear, clamp extrapolation")
    @printf(
        "%-21s %9s │ %9s %9s │ %9s %9s │ %9s %9s │ %9s\n",
        "case", "nq", "FI CPU", "Itp CPU", "FI kern", "Itp kern", "FI wall", "Itp wall", "|FI-Itp|"
    )
    for (cname, grids) in nd_cases(), nq in NQS
        N = length(grids)
        A = field(grids)
        fi = linear_interp(grids, A; extrap = ClampExtrap())
        it = Itp.linear_interpolation(map(itp_axis, grids), A; extrapolation_bc = Itp.Flat())
        q = soa_queries(N, nq)
        out = Vector{T}(undef, nq)
        t_fi = @belapsed $fi($out, $q)
        t_it = @belapsed $out .= $it.($(q)...)
        dfi = adapt(MtlArray, fi)
        qd = map(MtlArray, q)
        outd = MtlArray(out)
        k_fi, w_fi = device_time(() -> dfi(outd, qd)), gpu_wall(() -> dfi(outd, qd))
        k_it, w_it = try
            dit = adapt(MtlArray, it)
            f = () -> (outd .= dit.(qd...))
            device_time(f), gpu_wall(f)
        catch err
            @warn "Interpolations.jl on Metal failed for $cname" exception = (err, catch_backtrace()) maxlog = 1
            NaN, NaN
        end
        d_pkg = maximum(abs, fi(q) .- it.(q...))
        @printf(
            "%-21s %9d │ %9.0f %9.0f │ %9.0f %9.0f │ %9.0f %9.0f │ %9.1e\n",
            cname, nq, us(t_fi), us(t_it), us(k_fi), us(k_it), us(w_fi), us(w_it), d_pkg
        )
    end
    return
end

run_sections(("nd" => nd, "vs_itp" => vs_itp))
