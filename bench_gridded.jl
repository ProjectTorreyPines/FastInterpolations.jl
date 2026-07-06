# Does the FI GriddedQuery path match the hand-rolled anchor baseline speed?
# All Float64, in the worktree env. Manual timing throughout (no BenchmarkTools dep).
import FastInterpolations as FI
using FastInterpolations: GriddedQuery

qpos(n, m) = (sf = n / m; off = 0.5 - 0.5 * sf; Float64[sf * j + off for j in 1:m])
@inline function lanchor(x, n)
    xc = x < 1 ? 1.0 : (x > n ? Float64(n) : x)
    il = min(floor(Int, xc), n - 1)
    return (il, il + 1, xc - il)
end

# hand-rolled baseline: pass1 dim2 blend + pass2 dim1 precomputed-anchor
function handrolled(A, m1, m2)
    n1, n2 = size(A)
    q2 = qpos(n2, m2); q1 = qpos(n1, m1)
    B = Matrix{Float64}(undef, n1, m2)
    @inbounds for j in 1:m2
        il, ir, β = lanchor(q2[j], n2); a = 1 - β
        @views @. B[:, j] = a * A[:, il] + β * A[:, ir]
    end
    il = Vector{Int}(undef, m1); ir = similar(il); α = Vector{Float64}(undef, m1)
    @inbounds for i in 1:m1
        l, r, a = lanchor(q1[i], n1); il[i] = l; ir[i] = r; α[i] = a
    end
    C = Matrix{Float64}(undef, m1, m2)
    @inbounds for j in 1:m2
        @simd for i in 1:m1
            a = α[i]; C[i, j] = (1 - a) * B[il[i], j] + a * B[ir[i], j]
        end
    end
    return C
end

function best(f, reps)
    f()  # warmup
    b = Inf
    for _ in 1:reps
        b = min(b, @elapsed f())
    end
    return b
end

ns(t, np) = lpad(round(t / np * 1.0e9; digits = 2), 7)

function report(tag, n, m; reps = 300)
    A = rand(n, n); np = m * m
    itp = FI.linear_interp((1:n, 1:n), A; extrap = FI.ClampExtrap(), store = FI.StorePolicy(; copy = false))
    tx = qpos(n, m); ty = qpos(n, m); gq = GriddedQuery((tx, ty))
    G = itp(gq); H = handrolled(A, m, m)
    d = maximum(abs.(G .- H))
    tG = best(() -> itp(gq), reps)
    tH = best(() -> handrolled(A, m, m), reps)
    tP = best(() -> [itp((tx[i], ty[j])) for i in 1:m, j in 1:m], max(reps ÷ 4, 30))
    return println(
        rpad(tag, 16),
        " gridded=", ns(tG, np), " handrolled=", ns(tH, np), " pointwise=", ns(tP, np),
        "  gridded/hand=", lpad(round(tG / tH; digits = 2), 5),
        " gridded/pointwise=", lpad(round(tG / tP; digits = 2), 5), "  Δ=", round(d; sigdigits = 2)
    )
end

println("FI GriddedQuery vs hand-rolled anchor vs point-wise tensor (Float64). ns/output-px")
println("arch=", Sys.ARCH, "  FI=", pkgversion(FI))
report("512->256 down", 512, 256)
report("256->512 up", 256, 512)
report("256->1024 up", 256, 1024)
report("512->768 up", 512, 768)

# ── calibration + order-model verification ─────────────────────────────────
import FastInterpolations as FI
using FastInterpolations: _axis_anchors, _pass_blend_dim2!, _pass_gather_dim1!,
    _gridded_dim2_first, LinearInterp, EvalValue

function _bench_passes(TF, n1, n2, M, N; reps = 50)
    A = rand(TF, n1, n2)
    itp = FI.linear_interp((TF.(1:n1), TF.(1:n2)), A; extrap = FI.ClampExtrap())
    tx = collect(range(TF(1), TF(n1), M)); ty = collect(range(TF(1), TF(n2), N))
    p1 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[1], tx, itp.extraps[1], 1)
    p2 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[2], ty, itp.extraps[2], 2)
    BA = Matrix{TF}(undef, n1, N); CA = Matrix{TF}(undef, M, N)
    BB = Matrix{TF}(undef, M, n2); CB = Matrix{TF}(undef, M, N)
    tA = best(() -> (_pass_blend_dim2!(BA, itp.data, p2); _pass_gather_dim1!(CA, BA, p1)), reps)
    tB = best(() -> (_pass_gather_dim1!(BB, itp.data, p1); _pass_blend_dim2!(CB, BB, p2)), reps)
    # per-element pass costs for calibration (blend cost from order A pass 1)
    t_blend = best(() -> _pass_blend_dim2!(BA, itp.data, p2), reps)
    t_gath = best(() -> _pass_gather_dim1!(CA, BA, p1), reps)
    c_blend = t_blend * 1.0e9 / (n1 * N)
    c_gath = t_gath * 1.0e9 / (M * N)
    model_a = _gridded_dim2_first(n1, n2, M, N)
    winner_a = tA <= tB
    agree = model_a == winner_a || abs(tA - tB) / min(tA, tB) < 0.1   # within-noise tolerance
    println(
        "$TF $(n1)x$(n2) -> $(M)x$(N) : A=$(round(tA * 1.0e6, digits = 1))µs B=$(round(tB * 1.0e6, digits = 1))µs ",
        "model=$(model_a ? "A" : "B") measured=$(winner_a ? "A" : "B") agree=$agree ",
        "c_blend=$(round(c_blend, digits = 2)) c_gather=$(round(c_gath, digits = 2))"
    )
    return agree
end

println("\n== order-model verification (agree must be true everywhere) ==")
ok = true
for TF in (Float64, Float32)
    global ok &= _bench_passes(TF, 512, 512, 256, 256)   # down
    global ok &= _bench_passes(TF, 256, 256, 512, 512)   # up
    global ok &= _bench_passes(TF, 256, 256, 1024, 1024) # strong up
    global ok &= _bench_passes(TF, 512, 512, 768, 768)   # mild up
    global ok &= _bench_passes(TF, 512, 512, 1024, 64)   # mixed up/down
end
println(ok ? "MODEL OK" : "MODEL MISPICK — recalibrate _GRIDDED_C_* from the printed c_ values")
