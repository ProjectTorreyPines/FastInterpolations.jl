# Same-process A/B: lean payload-anchor Series batch (shipped entry) vs the
# pre-migration full-anchor recipe (its building blocks are retained verbatim:
# `_fill_anchors!` + `_eval_series_vector!`). Same process ⇒ no cross-env drift.
#
# Run: julia --project=benchmark benchmark/cubic_series_payload_ab.jl
# Env: SEC (per-benchmark budget, default 0.7), N (grid pts, default 256),
#      Q (queries, default 4096)

using FastInterpolations
const FI = FastInterpolations
using BenchmarkTools
using Random
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = parse(Float64, get(ENV, "SEC", "0.7"))
Random.seed!(20260710)

const N = parse(Int, get(ENV, "N", "256"))
const Q = parse(Int, get(ENV, "Q", "4096"))

const x = collect(range(0.0, 1.0, N))

# ── old arm: exact pre-migration batch recipe (full anchors, prealloc'd buffer —
# no pool bookkeeping, i.e. a slightly FLATTERED baseline; wins vs it are real).
function old_batch!(outputs, aq_vec, sitp, xq, op)
    searcher = FI._resolve_search(sitp.cache.x, xq, sitp.search_policy, nothing)
    FI._fill_anchors!(aq_vec, sitp.cache.x, xq, Val(:cubic), FI._should_wrap(sitp), searcher)
    y, z = sitp.y, sitp.z
    n_pts = FI.n_points(sitp)
    Tg = eltype(sitp.cache.x)
    x_min, x_max = Tg(first(sitp.cache.x)), Tg(last(sitp.cache.x))
    extrap = sitp.extrap
    @inbounds for k in 1:FI.n_series(sitp)
        FI._eval_series_vector!(outputs[k], y, z, n_pts, x_min, x_max, k, aq_vec, extrap, op)
    end
    return outputs
end

new_batch!(outputs, sitp, xq, op) = sitp(outputs, xq; deriv = op)

function oob_queries(frac, mode)
    n_oob = round(Int, frac * Q)
    xq = 0.02 .+ 0.96 .* rand(Q)
    if mode === :clustered
        xq[1:n_oob] .= -0.5
    else
        idx = randperm(Q)[1:n_oob]
        xq[idx] .= -0.5
    end
    return xq
end

results = Vector{Tuple{String, Float64, Float64}}()

function ab!(key, sitp, xq, op, Tq_w, I)
    K = FI.n_series(sitp)
    outputs = [Vector{Float64}(undef, Q) for _ in 1:K]
    Tg = eltype(sitp.cache.x)
    aq_vec = Vector{FI._CubicAnchoredQuery{Tg, Tq_w, I}}(undef, Q)
    t_old = @belapsed old_batch!($outputs, $aq_vec, $sitp, $xq, $op)
    t_new = @belapsed new_batch!($outputs, $sitp, $xq, $op)
    push!(results, (key, 1.0e6 * t_old, 1.0e6 * t_new))
    @printf("RESULT %-28s old %9.1f us   new %9.1f us   ratio %5.3f\n", key, 1.0e6 * t_old, 1.0e6 * t_new, t_new / t_old)
    return nothing
end

println("── K×Q sweep (in-domain), N=$N Q=$Q ──")
for K in (2, 8, 64)
    ys = [rand(N) for _ in 1:K]
    xq = 0.02 .+ 0.96 .* rand(Q)
    I = FI._interval_type(x)
    for (ename, extrap) in (("ext", ExtendExtrap()), ("clamp", ClampExtrap()))
        sitp = cubic_interp(x, Series(ys...); extrap = extrap)
        for (oname, op) in (("value", EvalValue()), ("deriv1", DerivOp(1)), ("deriv2", DerivOp(2)))
            ab!("K$(K)/$(ename)/$(oname)", sitp, xq, op, Float64, I)
        end
    end
end

println("── Clamp OOB-fraction sweep (K=8, value) ──")
let K = 8
    ys = [rand(N) for _ in 1:K]
    sitp = cubic_interp(x, Series(ys...); extrap = ClampExtrap())
    I = FI._interval_type(x)
    for frac in (0.0, 0.01, 0.1, 0.5, 1.0), mode in (:random, :clustered)
        xq = oob_queries(frac, mode)
        ab!("oob$(frac)/$(mode)", sitp, xq, EvalValue(), Float64, I)
    end
end

println("── one-shot vector batch (K=8, value/deriv2, Extend) ──")
let K = 8
    ys = [rand(N) for _ in 1:K]
    s = Series(ys...)
    xq = 0.02 .+ 0.96 .* rand(Q)
    outs = [Vector{Float64}(undef, Q) for _ in 1:K]
    for (oname, op) in (("value", EvalValue()), ("deriv2", DerivOp(2)))
        t = @belapsed cubic_interp!($outs, $x, $s, $xq; extrap = ExtendExtrap(), deriv = $op)
        @printf("RESULT oneshot/%-19s new %9.1f us  (no in-process old arm; cross-check vs base branch if needed)\n", oname, 1.0e6 * t)
    end
end

println("── scalar sanity (unchanged path, K=8) ──")
let K = 8
    ys = [rand(N) for _ in 1:K]
    sitp = cubic_interp(x, Series(ys...); extrap = ClampExtrap())
    t = @belapsed $sitp(0.5)
    @printf("RESULT scalar/value              %9.3f us\n", 1.0e6 * t)
end

println("\n── summary (new/old ratio < 1.0 = lean anchors faster) ──")
for (key, t_old, t_new) in results
    @printf("%-28s %6.3f\n", key, t_new / t_old)
end
