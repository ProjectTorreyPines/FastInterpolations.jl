# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║     ND vector calculus — ∂/∂data via Enzyme for vector-valued data         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Loss = f(s(·)) where s collapses the channels linearly (s(v) = Σ cₖ vₖ). By
# linearity of interpolation in the data, channel k of ∂data equals cₖ times the
# Float64 ∂data of the same loss on the combined data D = Σ cₖ·channelₖ.
#
# Top-level loss functions over `const` globals: Enzyme's IR checks are most stable
# without closures over non-const values.

using Test
using FastInterpolations
using StaticArrays

const _EV_SKIP = Sys.iswindows() && VERSION >= v"1.12"      # Enzyme LLVM issues (see test_autodiff_Enzyme.jl)
const _EV_SUPPORTED = VERSION >= v"1.11"                    # ND struct API rules need ≥ 1.11

if !_EV_SKIP && _EV_SUPPORTED
    using Enzyme

    const _EV_X = range(0.0, 2.0, 11)
    const _EV_Y = range(0.0, 1.0, 9)
    const _EV_Q = (0.7, 0.4)
    _ev_itp(d) = cubic_interp((_EV_X, _EV_Y), d)

    _ev_s(v::Real) = v
    _ev_s(v::SVector{3}) = v[1] + 2 * v[2] + 3 * v[3]
    _ev_s(v::Complex) = real(v) + 2 * imag(v)
    _ev_col(J::StaticMatrix, j) = J[:, j]
    _ev_col(J::AbstractMatrix, j) = J[1, j]

    _ev_eval(d) = _ev_s(_ev_itp(d)(_EV_Q))^2
    _ev_grad(d) = (g = FastInterpolations.gradient(_ev_itp(d), _EV_Q); _ev_s(g[1])^2 + _ev_s(g[2]))
    _ev_jac(d) = (J = FastInterpolations.jacobian(_ev_itp(d), _EV_Q); _ev_s(_ev_col(J, 1))^2 + _ev_s(_ev_col(J, 2)))
    _ev_hess(d) = (H = FastInterpolations.hessian(_ev_itp(d), _EV_Q); _ev_s(H[1, 1])^2 + _ev_s(H[1, 2]) + _ev_s(H[2, 2])^2)
    _ev_lap(d) = _ev_s(FastInterpolations.laplacian(_ev_itp(d), _EV_Q))^2
    _ev_oneshot(d) = _ev_s(cubic_interp((_EV_X, _EV_Y), d, _EV_Q))^2

    # Independent oracle for the Float64 reference: central differences in every entry.
    function _ev_fd(f, D; h = 1.0e-6)
        g = similar(D)
        for i in eachindex(D)
            Dp = copy(D); Dp[i] += h
            Dm = copy(D); Dm[i] -= h
            g[i] = (f(Dp) - f(Dm)) / (2h)
        end
        return g
    end

    @testset "ND vector calculus — vector-valued data, ∂/∂data via Enzyme" begin
        C = (
            [sin(a) * cos(b) for a in _EV_X, b in _EV_Y],
            [a * b + 0.3 * a for a in _EV_X, b in _EV_Y],
            [cos(a + b) for a in _EV_X, b in _EV_Y],
        )
        # (label, s coefficients, value from channels, channel k of a ∂data entry, eval error today)
        kinds = (
            ("SVector{3}", (1.0, 2.0, 3.0), c -> SVector(c...), (e, k) -> e[k], DimensionMismatch),
            ("ComplexF64", (1.0, 2.0), c -> complex(c...), (e, k) -> k == 1 ? real(e) : imag(e), MethodError),
        )
        # (label, loss, Float64 oracle loss) — the jacobian loss equals the gradient loss
        ops = (
            ("gradient", _ev_grad, _ev_grad),
            ("jacobian", _ev_jac, _ev_grad),
            ("hessian", _ev_hess, _ev_hess),
            ("laplacian", _ev_lap, _ev_lap),
            ("one-shot eval", _ev_oneshot, _ev_oneshot),
        )
        for (klabel, coef, mk, chan, eval_error) in kinds
            M = length(coef)
            data = [mk(ntuple(k -> C[k][i], M)) for i in CartesianIndices(C[1])]
            D = sum(coef[k] .* C[k] for k in 1:M)
            for (olabel, f, fref) in ops
                @testset "$klabel | $olabel" begin
                    dd = Enzyme.make_zero(data)
                    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(data, dd))
                    dD = zero(D)
                    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(fref), Enzyme.Active, Enzyme.Duplicated(D, dD))
                    @test dD ≈ _ev_fd(fref, D) rtol = 1.0e-6 atol = 1.0e-8
                    for k in 1:M
                        @test map(e -> chan(e, k), dd) ≈ coef[k] .* dD rtol = 1.0e-10 atol = 1.0e-12
                    end
                end
            end
            # `itp(q)` with vector-valued data is not supported under Enzyme yet (the
            # callable-object rule receives a zero cotangent for a multi-component return);
            # it must fail loudly, never return a silently zero gradient.
            @testset "$klabel | eval refuses" begin
                dd = Enzyme.make_zero(data)
                @test_throws eval_error Enzyme.autodiff(
                    Enzyme.Reverse, Enzyme.Const(_ev_eval), Enzyme.Active, Enzyme.Duplicated(data, dd)
                )
            end
        end
    end
end
