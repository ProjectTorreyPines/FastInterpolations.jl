# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║               LINEAR 1D — DUCK-TYPED GRID SCALAR (Phase 1)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Structural verification that LinearInterpolant accepts a homogeneous duck-typed
# grid scalar, following the same pattern Tv duck-typing uses.
#
# Uses a minimal local `TDG` (Test Dual Grid) type that mimics ForwardDiff.Dual
# semantics (primal + single partial) without pulling in ForwardDiff. This lets
# us exercise the `else`-branch in `_promote_itp_inputs`, the duck path in
# `_create_spacing`, and the unified `linear_interp` scalar API without any
# extension loaded.
#
# The actual ForwardDiff.Dual integration tests live in
# `test/ext/test_linear_dual_grid.jl` (runs under the extension test entrypoint).

using Test
using FastInterpolations

const FI = FastInterpolations

# ─── Minimal duck-typed grid scalar (1-component forward-mode dual) ──────────
#
# Subtypes `Real` so that `_to_grid_type(xq::Real, ::Type{TDG})` dispatches to
# the existing `Tg(_extract_primal(xq))` fallback in utils.jl, which calls the
# `TDG(::Real)` constructor we provide.
struct TDG <: Real
    v::Float64   # primal value
    d::Float64   # partial derivative w.r.t. the synthetic parameter
end

# Float → TDG lift (zero partial): needed by _to_grid_type in _search_binary.
TDG(x::Real) = TDG(Float64(x), 0.0)

# Forward-mode AD rules on arithmetic
Base.:-(a::TDG, b::TDG) = TDG(a.v - b.v, a.d - b.d)
Base.:+(a::TDG, b::TDG) = TDG(a.v + b.v, a.d + b.d)
Base.:*(a::TDG, b::TDG) = TDG(a.v * b.v, a.v * b.d + a.d * b.v)
Base.:/(a::TDG, b::TDG) = a * inv(b)
Base.inv(a::TDG) = (q = inv(a.v); TDG(q, -a.d * q * q))
Base.:-(a::TDG) = TDG(-a.v, -a.d)

# Mixed TDG/Real arithmetic (kernel path uses Float y values)
Base.:+(a::TDG, b::Real) = TDG(a.v + b, a.d)
Base.:+(a::Real, b::TDG) = TDG(a + b.v, b.d)
Base.:-(a::TDG, b::Real) = TDG(a.v - b, a.d)
Base.:-(a::Real, b::TDG) = TDG(a - b.v, -b.d)
Base.:*(a::TDG, b::Real) = TDG(a.v * b, a.d * b)
Base.:*(a::Real, b::TDG) = TDG(a * b.v, a * b.d)
Base.:/(a::TDG, b::Real) = TDG(a.v / b, a.d / b)

# zero/one (used by kernel zero-return branches and promotion defaults)
Base.zero(::Type{TDG}) = TDG(0.0, 0.0)
Base.zero(::TDG) = TDG(0.0, 0.0)
Base.one(::Type{TDG}) = TDG(1.0, 0.0)
Base.one(::TDG) = TDG(1.0, 0.0)

# Ordering: forward to primal (grid sortedness is a primal concept)
Base.:<(a::TDG, b::TDG) = a.v < b.v
Base.:<(a::TDG, b::Real) = a.v < b
Base.:<(a::Real, b::TDG) = a < b.v
Base.:<=(a::TDG, b::TDG) = a.v <= b.v
Base.:<=(a::TDG, b::Real) = a.v <= b
Base.:<=(a::Real, b::TDG) = a <= b.v
Base.isless(a::TDG, b::TDG) = a.v < b.v
Base.isless(a::TDG, b::Real) = a.v < b
Base.isless(a::Real, b::TDG) = a < b.v

# Promotion: TDG + Real → TDG (keeps Real widenings flowing to TDG)
Base.promote_rule(::Type{TDG}, ::Type{<:Real}) = TDG

# float(): required by _promote_grid_float. TDG is already "float-like" (wraps Float64).
Base.float(::Type{TDG}) = TDG
Base.float(x::TDG) = x

# isapprox for test comparisons on the primal field
approx_primal(a::TDG, v::Real; atol = 1.0e-12) = isapprox(a.v, v; atol = atol)
approx_partial(a::TDG, d::Real; atol = 1.0e-12) = isapprox(a.d, d; atol = atol)

# ─── Tests ───────────────────────────────────────────────────────────────────

@testset "Linear 1D — duck-typed grid (TDG)" begin

    @testset "Construction with Vector{TDG}" begin
        # Grid "shifts uniformly" with synthetic parameter: ∂x[i]/∂ε = 1 for all i.
        x = [TDG(1.0, 1.0), TDG(2.0, 1.0), TDG(3.0, 1.0)]
        y = [10.0, 20.0, 15.0]

        itp = linear_interp(x, y)

        # Struct type parameters propagate the duck type correctly
        @test itp isa FI.LinearInterpolant
        @test eltype(itp.x) === TDG
        @test eltype(itp.y) === Float64

        # spacing is a real VectorSpacing{TDG} — h and inv_h cached as TDG
        @test itp.spacing isa FI.VectorSpacing{TDG}
        @test length(itp.spacing.h) == 2
        @test length(itp.spacing.inv_h) == 2
        # h[1] primal = 1, partial = 0 (both grid ends shift by 1, so diff is 0)
        @test approx_primal(itp.spacing.h[1], 1.0)
        @test approx_partial(itp.spacing.h[1], 0.0)
        @test approx_primal(itp.spacing.inv_h[1], 1.0)
        @test approx_partial(itp.spacing.inv_h[1], 0.0)
    end

    @testset "Scalar callable returns TDG with correct derivative" begin
        # Setup: uniform shift grid, linear-in-x data on first segment.
        x = [TDG(1.0, 1.0), TDG(2.0, 1.0), TDG(3.0, 1.0)]
        y = [10.0, 20.0, 15.0]
        itp = linear_interp(x, y)

        # Mid-interval query in [x[1], x[2]]:
        # α = (xq − xL)/(xR − xL) = (1.5 − 1)/(2 − 1) = 0.5
        # value = y[1] + α*(y[2]−y[1]) = 10 + 0.5*10 = 15
        # ∂α/∂ε = ∂((1.5−1−ε)/(1))/∂ε = −1
        # ∂value/∂ε = −1 * (y[2]−y[1]) = −10
        r = itp(1.5)
        @test r isa TDG
        @test approx_primal(r, 15.0)
        @test approx_partial(r, -10.0)

        # Second interval: query at 2.5, in [x[2], x[3]]
        # α = (2.5−2)/(3−2) = 0.5
        # value = 20 + 0.5*(15−20) = 17.5
        # ∂α/∂ε = −1 (same reasoning, uniform shift)
        # ∂value/∂ε = −1 * (15−20) = 5
        r2 = itp(2.5)
        @test approx_primal(r2, 17.5)
        @test approx_partial(r2, 5.0)
    end

    @testset "One-shot scalar matches callable form" begin
        x = [TDG(0.0, 1.0), TDG(1.0, 1.0), TDG(2.0, 1.0), TDG(3.0, 1.0)]
        y = [0.0, 1.0, 4.0, 9.0]
        itp = linear_interp(x, y)
        for xq in (0.25, 0.9, 1.5, 2.75)
            a = itp(xq)
            b = linear_interp(x, y, xq)
            @test a.v ≈ b.v
            @test a.d ≈ b.d
        end
    end

    @testset "Endpoint queries" begin
        x = [TDG(0.0, 0.0), TDG(1.0, 1.0), TDG(2.0, 2.0)]
        y = [5.0, 7.0, 11.0]
        itp = linear_interp(x, y)

        # Query exactly at x[1] (which has ∂=0): answer is y[1] with partial = 0
        r_lo = itp(0.0)
        @test approx_primal(r_lo, 5.0)
        @test approx_partial(r_lo, 0.0)

        # Query exactly at x[end] lies on the [x[2], x[3]] boundary.
        # Binary search picks the last interval; dL = xq − xL = 2 − (1+ε*1) scaled.
        # At primal: (2 − 1)/(2 − 1) = 1 → value = y[3]
        # This is the non-differentiable knot case; we only assert the primal.
        r_hi = itp(2.0)
        @test approx_primal(r_hi, 11.0)
    end

    @testset "Non-uniform grid + non-trivial partial recovery" begin
        # Only x[2] is "sensitive" to the parameter; x[1], x[3] are fixed.
        x = [TDG(0.0, 0.0), TDG(1.0, 2.0), TDG(3.0, 0.0)]
        y = [0.0, 10.0, 20.0]
        itp = linear_interp(x, y)

        # Query at 2.0 → interval [x[2], x[3]] = [TDG(1,2), TDG(3,0)]
        # Primal α = (2 − 1)/(3 − 1) = 0.5
        # ε-analytical derivative (quotient rule):
        #   xL(ε) = 1 + 2ε, xR(ε) = 3
        #   α(ε)  = (2 − 1 − 2ε) / (3 − 1 − 2ε) = (1 − 2ε)/(2 − 2ε)
        #   dα/dε|0 = (−2*2 − 1*(−2))/4 = (−4 + 2)/4 = −0.5
        #   d(value)/dε = dα/dε * (y[3] − y[2]) = −0.5 * 10 = −5
        r = itp(2.0)
        @test approx_primal(r, 15.0)
        @test approx_partial(r, -5.0)
    end

    @testset "Struct types carry the duck type everywhere" begin
        x = [TDG(float(i), 0.0) for i in 1:5]
        y = rand(5)
        itp = linear_interp(x, y)

        # Grid-side type parameters
        @test itp isa FI.LinearInterpolant{TDG, Float64}
        @test typeof(itp.x) <: AbstractVector{TDG}
        @test typeof(itp.spacing) === FI.VectorSpacing{TDG}
    end

    @testset "Regression: Float grid path still zero-alloc" begin
        # Verify the duck-type relaxation did not disturb the Float fast path.
        x = collect(1.0:0.1:10.0)
        y = sin.(x)
        itp = linear_interp(x, y)

        # Function barrier + warmup
        function measure_float_callable()
            itp_local = linear_interp(x, y)
            itp_local(5.55)  # warmup
            return @allocations itp_local(5.55)
        end
        @test measure_float_callable() == 0
    end

    @testset "Regression: Int grid → Float auto-promotion still works" begin
        # Suppress the one-shot `_to_float` warning for the Int grid case.
        x_int = collect(1:10)
        y_int = (1:10) .^ 2
        # Callable path: constructor promotes via _promote_itp_inputs
        itp_int = linear_interp(x_int, y_int)
        @test eltype(itp_int.x) === Float64
        @test eltype(itp_int.y) === Float64
        @test itp_int(4.5) ≈ 0.5 * (16 + 25)  # midpoint between 4² and 5²
    end
end
