# CPU vs Metal performance of 1D LinearInterpolant (Float32, n = 1001; informational, not a
# public claim). Sections (all by default, or pick by name):
#   breakdown : where a GPU batch call spends its time, next to the CPU batch call
#   vs_itp    : FastInterpolations vs Interpolations.jl, same grid / data / queries, CPU and Metal
#
#   julia --project=scripts/gpu scripts/gpu/bench_linear1d.jl [breakdown] [vs_itp]
#
# Queries: seeded random OOB mix on [-0.2, 1.2) with clamp extrapolation (FI `ClampExtrap()`
# ≡ Interpolations `Flat()`); "sorted" sorts the same queries. All times µs, minimum.

using FastInterpolations, Adapt, GPUArrays, Metal
import Interpolations
const Itp = Interpolations
include(joinpath(@__DIR__, "..", "..", "test", "gpu", "linear1d_cases.jl"))
include(joinpath(@__DIR__, "metal_timing.jl"))

const T = Float32
const N_GRID = 1001
const NQS = (10^4, 10^6, 10^7)
const GPUExt = Base.get_extension(FastInterpolations, :FastInterpolationsGPUArraysExt)
GPUArrays.allowscalar(false)

function bench_cases(nq)
    q = rand_queries(T, nq)[2]
    xr, xv = last.(gpu_grids(T, N_GRID))
    return (("Range/random", xr, q), ("Vector/random", xv, q), ("Vector/sorted", xv, sort(q)))
end

# CPU eval    in-place `itp(out, q)` (single thread, 0 allocs)
# GPU kern    device time of the evaluation (pure on-device execution)
# launch+sync wall of `ditp(out_d, q_d); synchronize()` minus GPU kern — two minima from
#             separate windows; Metal's GPU performance state shifts over seconds, so this
#             difference is noisy for mid-size cases and can come out negative
# NoExt chk   NoExtrap batch domain check (one fused device reduction + readback)
# H→D / D→H   query upload / result download, timed on their own
# copy kern   device time of `out_d .= q_d` (bandwidth reference, same bytes moved)
function breakdown()
    println("CPU vs Metal breakdown — ClampExtrap, n = $N_GRID")
    @printf(
        "%-14s%10s%10s%10s%9s%12s%11s%9s%9s%10s\n",
        "case", "nq", "CPU eval", "GPU kern", "CPU/GPU", "launch+sync", "NoExt chk", "H→D", "D→H", "copy kern"
    )
    for nq in NQS
        qd_in = MtlArray(rand_queries(T, nq)[1])           # the NoExtrap check needs in-domain queries
        for (cname, x, q) in bench_cases(nq)
            itp = linear_interp(x, gpu_fy.(x); extrap = ClampExtrap())
            ditp = adapt(MtlArray, itp)
            out = similar(q)
            qd, outd = MtlArray(q), MtlArray(out)
            itp(out, q)
            @assert (@allocated itp(out, q)) == 0
            t_cpu = @belapsed $itp($out, $q)
            t_kern = device_time(() -> ditp(outd, qd))
            t_wall = gpu_wall(() -> ditp(outd, qd))
            t_chk = @belapsed $(GPUExt._gpu_domain_check)($(ditp.x), $qd_in, NoExtrap())
            t_h2d = gpu_wall(() -> copyto!(qd, q))
            t_d2h = @belapsed copyto!($out, $outd)
            t_copy = device_time(() -> (outd .= qd))
            @printf(
                "%-14s%10d%10.1f%10.1f%8.1fx%12.1f%11.1f%9.0f%9.0f%10.1f\n", cname, nq, us(t_cpu), us(t_kern),
                t_cpu / t_kern, us(t_wall - t_kern), us(t_chk), us(t_h2d), us(t_d2h), us(t_copy)
            )
        end
    end
    return
end

# CPU   FI batch `itp(out, q)`; FI and Itp broadcast `out .= f.(q)`
# GPU   kern = device time of `out_d .= f.(q_d)`; wall = launch + execution + sync, no copies
# Itp gets a `LinRange` for the uniform grid: a Base `range(0f0, 1f0; length)` keeps Float64
# internals that Metal cannot compile ("unsupported use of double value"); FI converts any
# range to its Float32 `_CachedRange`.
function vs_itp()
    println("FastInterpolations (FI) vs Interpolations.jl (Itp) — clamp extrapolation, n = $N_GRID")
    @printf(
        "%-14s%10s │%9s%9s%9s │%9s%9s │%9s%9s%9s │%10s%10s\n", "case", "nq",
        "FI batch", "FI bcast", "Itp bcast", "FI kern", "Itp kern", "FI wall", "FIbatch w", "Itp wall", "|FI-Itp|", "|GPU-CPU|"
    )
    for nq in NQS, (cname, x, q) in bench_cases(nq)
        y = gpu_fy.(x)
        fi = linear_interp(x, y; extrap = ClampExtrap())
        it = Itp.linear_interpolation(x isa AbstractRange ? LinRange(first(x), last(x), length(x)) : x, y; extrapolation_bc = Itp.Flat())
        dfi, dit = adapt(MtlArray, fi), adapt(MtlArray, it)
        out = similar(q)
        qd = MtlArray(q)
        outd = similar(qd)
        t_fi_b = @belapsed $fi($out, $q)
        t_fi_bc = @belapsed $out .= $fi.($q)
        t_it_bc = @belapsed $out .= $it.($q)
        k_fi = device_time(() -> (outd .= dfi.(qd)))
        k_it = device_time(() -> (outd .= dit.(qd)))
        w_fi = gpu_wall(() -> (outd .= dfi.(qd)))
        w_fib = gpu_wall(() -> dfi(outd, qd))
        w_it = gpu_wall(() -> (outd .= dit.(qd)))
        d_pkg = maximum(abs, fi.(q) .- it.(q))
        d_gpu = maximum(abs, Array(dfi.(qd)) .- fi.(q))
        @printf(
            "%-14s%10d │%9.0f%9.0f%9.0f │%9.0f%9.0f │%9.0f%9.0f%9.0f │%10.1e%10.1e\n", cname, nq,
            us(t_fi_b), us(t_fi_bc), us(t_it_bc), us(k_fi), us(k_it), us(w_fi), us(w_fib), us(w_it), d_pkg, d_gpu
        )
    end
    return
end

run_sections(("breakdown" => breakdown, "vs_itp" => vs_itp))
