# Does the FI GriddedQuery path match the hand-rolled anchor baseline speed?
# All Float64, in the worktree env. Manual timing (no BenchmarkTools dep).
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
