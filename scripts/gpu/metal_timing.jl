# Shared helpers for the local Metal benchmark scripts (e.g. bench_linear1d.jl).
# Not used by the CI test, which has neither Metal nor BenchmarkTools.
#   CPU / GPU wall : BenchmarkTools minimum (`f(); Metal.synchronize()` on the GPU)
#   GPU device     : minimum over repetitions of the summed command-buffer time
#                    (GPUEndTime − GPUStartTime) of `f` — pure on-device execution
# Metal's GPU performance state changes over seconds (not fixed by a warm-up), so the same
# case can land on different levels across calls (~2×); compare within one run, and do not
# subtract device from wall times measured in different windows without that caveat.

using BenchmarkTools, Metal, Printf, Random

us(t) = t * 1.0e6
ms(t) = t * 1.0e3

function device_time(f; reps = 20)
    f()
    Metal.synchronize()
    best = Inf
    for _ in 1:reps
        d = Metal.Profiling.profile_internally(f).device
        isempty(d.start) || (best = min(best, sum(d.stop .- d.start)))
    end
    return best
end

gpu_wall(f; seconds = 5) = @belapsed ($f(); Metal.synchronize()) seconds = seconds

# Seeded uniform random queries: in-domain [0, 1) and an OOB mix on [-0.2, 1.2). Branchy CPU
# loops need these: the golden-ratio sequence of `gpu_queries` is learned by the branch
# predictor and flatters the CPU.
function rand_queries(::Type{T}, nq::Int; seed::Int = 7) where {T}
    q_in = rand(Xoshiro(seed), T, nq)
    return q_in, T(1.4) .* q_in .- T(0.2)
end

# Run the sections named in ARGS (all when none), in declaration order.
function run_sections(sections)
    names = first.(sections)
    unknown = setdiff(ARGS, names)
    isempty(unknown) || error("unknown section(s) $(join(unknown, ", ")); available: $(join(names, ", "))")
    for (name, f) in sections
        isempty(ARGS) || name in ARGS || continue
        println("\n════ ", name, " ════")
        f()
    end
    return nothing
end
