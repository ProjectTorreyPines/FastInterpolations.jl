# Same-process runtime-swap A/B for the componentwise colorant blend.
#   A = componentwise (ext entries active)   B = generic (entries swapped by @eval)
# Public scalar path, 1D + 2D UnitStep grids (1:16), InBounds — the real kernel
# shape (matches claudedocs/scratch/color_component_actual_shape_bench.jl).
# Interleaved rounds; report min ns/eval. Float64/ComplexF64 are controls
# (path untouched — must be flat and bit-identical across the swap).
using BenchmarkTools
using ColorTypes
using ColorVectorSpace
using FixedPointNumbers
using FastInterpolations
import FastInterpolations as FI
using FastInterpolations: InBounds, linear_interp
using Random

const N = 4096
Random.seed!(42)
const Q1 = 1.0 .+ 14.998 .* rand(N)
const QX = 1.0 .+ 14.998 .* rand(N)
const QY = 1.0 .+ 14.998 .* rand(N)

randch() = 0.1 + 0.8 * rand()
randc(::Type{Gray{T}}) where {T} = Gray{T}(randch())
randc(::Type{RGB{T}}) where {T} = RGB{T}(randch(), randch(), randch())
randc(::Type{RGBA{T}}) where {T} = RGBA{T}(randch(), randch(), randch(), randch())
randc(::Type{AGray{T}}) where {T} = AGray{T}(randch(), randch())
randc(::Type{GrayA{T}}) where {T} = GrayA{T}(randch(), randch())
randc(::Type{ARGB{T}}) where {T} = ARGB{T}(randch(), randch(), randch(), randch())
randc(::Type{Float64}) = randch()
randc(::Type{ComplexF64}) = randch() + randch() * im

function run1d!(out, itp, q)
    @inbounds for i in eachindex(out)
        out[i] = itp(q[i])
    end
    return out
end
function run2d!(out, itp, qx, qy)
    @inbounds for i in eachindex(out)
        out[i] = itp(qx[i], qy[i])
    end
    return out
end

function bench_family(::Type{V}) where {V}
    d1 = [randc(V) for _ in 1:16]
    d2 = [randc(V) for _ in 1:16, _ in 1:16]
    itp1 = linear_interp(1:16, d1; extrap = InBounds())
    itp2 = linear_interp((1:16, 1:16), d2; extrap = (InBounds(), InBounds()))
    out1 = [itp1(Q1[1]) for _ in 1:N]
    out2 = [itp2(QX[1], QY[1]) for _ in 1:N]
    t1 = @benchmark run1d!($out1, $itp1, $Q1) evals = 1 samples = 1000 seconds = 2
    t2 = @benchmark run2d!($out2, $itp2, $QX, $QY) evals = 1 samples = 1000 seconds = 2
    return minimum(t1).time / N, minimum(t2).time / N
end

# ── A/B toggle: pin the family via an entry-level method from Main ──────────
# The shipped path styles colorants through the core rule (the ext adds a
# componentwise style — no entry override). Defining a family-scoped ENTRY
# method here outranks the core duck entry, pinning the family to one body
# regardless of the shipped rule: componentwise (mapc, α preserved per
# channel) vs the core generic styled escape. The cw form here is ungated
# (benchmark uses eligible channels only).
gate_expr(F) = quote
    @inline FI._linear_value_blend(α, yL::C, yR::C) where {C <: $F} =
        mapc((l, r) -> FI._linear_value_blend(α, l, r), yL, yR)
end
generic_expr(F) = quote
    @inline FI._linear_value_blend(α, yL::C, yR::C) where {C <: $F} =
        FI._linear_value_blend(FI._LinearBlendGeneric(), α, yL, yR)
end

const FAMILY_SYMS = (:Gray, :RGB, :RGBA, :AGray, :GrayA, :ARGB)
const FAMILY_TYPES = Dict(
    :Gray => Gray, :RGB => RGB, :RGBA => RGBA,
    :AGray => AGray, :GrayA => GrayA, :ARGB => ARGB,
)

function swap!(family::Symbol, mode::Symbol)
    F = nameof(FAMILY_TYPES[family])
    Core.eval(Main, mode === :generic ? generic_expr(F) : gate_expr(F))
    return nothing
end

const CASES = [
    (family = :Gray, V = Gray{N0f8}), (family = :Gray, V = Gray{Float64}),
    (family = :RGB, V = RGB{N0f8}), (family = :RGB, V = RGB{Float64}),
    (family = :RGBA, V = RGBA{N0f8}), (family = :RGBA, V = RGBA{Float64}),
    (family = :AGray, V = AGray{N0f8}), (family = :AGray, V = AGray{Float64}),
    (family = :GrayA, V = GrayA{N0f8}), (family = :GrayA, V = GrayA{Float64}),
    (family = :ARGB, V = ARGB{N0f8}), (family = :ARGB, V = ARGB{Float64}),
]

function main(; rounds::Int = 3)
    # correctness/identity control BEFORE timing: Float64 values must be
    # bit-identical across a swap (their path never consults the trait)
    dctl = [randc(Float64) for _ in 1:16]
    ictl = linear_interp(1:16, dctl; extrap = InBounds())
    vals_a = [Base.invokelatest(ictl, q) for q in Q1[1:64]]
    swap!(:Gray, :generic)
    vals_b = [Base.invokelatest(ictl, q) for q in Q1[1:64]]
    swap!(:Gray, :componentwise)
    @assert vals_a == vals_b "Float64 control drifted across style swap"
    @assert Base.invokelatest(FI._linear_blend_style, Float64, Gray{BigFloat}) ===
        FI._LinearBlendGeneric() "BigFloat channel must stay generic"

    results = Dict{Any, Vector{NTuple{2, Float64}}}()
    for round_i in 1:rounds, mode in (:componentwise, :generic)
        println(stderr, "── round $round_i / mode $mode")
        for c in CASES
            swap!(c.family, mode)
            t = Base.invokelatest(bench_family, c.V)
            push!(get!(results, (c.V, mode), NTuple{2, Float64}[]), t)
            swap!(c.family, :componentwise)
        end
        # controls each round (mode-independent path; keyed by mode to expose drift)
        for V in (Float64, ComplexF64)
            t = Base.invokelatest(bench_family, V)
            push!(get!(results, (V, mode), NTuple{2, Float64}[]), t)
        end
    end

    fmt(x) = string(round(x; digits = 2))
    println("| value type | mode | 1D min ns/eval | 2D min ns/eval |")
    println("| --- | --- | ---: | ---: |")
    for c in CASES, mode in (:componentwise, :generic)
        ts = results[(c.V, mode)]
        println(
            "| `", c.V, "` | ", mode, " | ",
            fmt(minimum(first, ts)), " | ", fmt(minimum(last, ts)), " |",
        )
    end
    for V in (Float64, ComplexF64), mode in (:componentwise, :generic)
        ts = results[(V, mode)]
        println(
            "| `", V, "` (control) | ", mode, " | ",
            fmt(minimum(first, ts)), " | ", fmt(minimum(last, ts)), " |",
        )
    end
    return results
end

main()
