# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║        ND reverse mode (Zygote) for vector-valued (and complex) data       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Interpolation is linear in the data, so a loss written channel by channel must give,
# per channel, the gradient of the same loss on Float64 twin interpolants (ForwardDiff
# oracle) — on the data side and on the query side.

using Test
using FastInterpolations
using Zygote
using ForwardDiff
using StaticArrays

# Channel `c` of a Jacobian: vector-valued data stacks channels as rows; otherwise each
# entry carries every channel.
_jac_channel(J::StaticMatrix, c, _) = J[c, :]
_jac_channel(J::AbstractMatrix, c, ch) = map(e -> ch(e, c), J)

# Losses: `itp_of(c)` is the interpolant holding channel `c`, `ch(v, c)` reads channel `c`
# of a value. Vector data: one interpolant, `ch` = the kind's accessor. Twin: one Float64
# interpolant per channel, `ch(v, c) = v`.
const VEC_W = (1.3, -0.7, 0.4)
const VEC_LOSSES = (
    ("eval", (itp_of, q, ch, M) -> sum(c -> VEC_W[c] * ch(itp_of(c)(q), c)^2, 1:M)),
    (
        "gradient", (itp_of, q, ch, M) -> sum(1:M) do c
            g = FastInterpolations.gradient(itp_of(c), q)
            VEC_W[c] * (ch(g[1], c)^2 - ch(g[2], c))
        end,
    ),
    (
        "value_gradient", (itp_of, q, ch, M) -> sum(1:M) do c
            v, g = FastInterpolations.value_gradient(itp_of(c), q)
            VEC_W[c] * (ch(v, c) + ch(g[1], c) * ch(g[2], c))
        end,
    ),
    (
        "jacobian", (itp_of, q, ch, M) -> sum(1:M) do c
            J = FastInterpolations.jacobian(itp_of(c), q)
            VEC_W[c] * sum(abs2, _jac_channel(J, c, ch))
        end,
    ),
    (
        "value_jacobian", (itp_of, q, ch, M) -> sum(1:M) do c
            v, J = FastInterpolations.value_jacobian(itp_of(c), q)
            VEC_W[c] * (ch(v, c)^2 + sum(_jac_channel(J, c, ch)))
        end,
    ),
    (
        "hessian", (itp_of, q, ch, M) -> sum(1:M) do c
            H = FastInterpolations.hessian(itp_of(c), q)
            VEC_W[c] * sum(abs2, map(h -> ch(h, c), H))
        end,
    ),
    ("laplacian", (itp_of, q, ch, M) -> sum(c -> VEC_W[c] * ch(FastInterpolations.laplacian(itp_of(c), q), c)^2, 1:M)),
)
_twin_ch(v, _) = v

@testset "ND reverse mode — vector-valued data (Zygote)" begin
    x = range(0.0, 2.0, 11)
    y = range(0.0, 1.0, 9)
    q0 = (0.7, 0.4)
    f1(a, b) = sin(π * a) * cos(b)
    f2(a, b) = cos(π * a) * b + 0.3
    f3(a, b) = sin(π * a + b)
    twins = Tuple([f(a, b) for a in x, b in y] for f in (f1, f2, f3))   # exactly periodic in x

    # (label, channels, value from channels, channel accessor; data-gradient entries use it too)
    kinds = (
        ("SVector{3}", 3, c -> SVector(c...), (v, c) -> v[c]),
        ("ComplexF64", 2, c -> complex(c...), (v, c) -> c == 1 ? real(v) : imag(v)),
    )
    # ∂query of hessian / laplacian needs third derivatives (not provided by their rules).
    query_ops = ("eval", "gradient", "value_gradient", "jacobian", "value_jacobian")
    builders = (
        ("cubic", cubic_interp, first.(VEC_LOSSES)),
        ("linear", linear_interp, ("eval", "gradient", "jacobian")),
        ("quadratic", quadratic_interp, ("eval", "gradient", "jacobian")),
        ("cubic PeriodicBC", (g, d) -> cubic_interp(g, d; bc = (PeriodicBC(), CubicFit())), ("eval", "jacobian")),
        ("hetero", (g, d) -> interp(g, d; method = (CubicInterp(), LinearInterp())), ("eval", "gradient", "jacobian")),
        ("constant", constant_interp, ("eval", "gradient")),
    )

    for (blabel, build, ops) in builders, (klabel, M, mk, ch) in kinds, (olabel, loss) in VEC_LOSSES
        olabel in ops || continue
        data = [mk(ntuple(c -> twins[c][i], M)) for i in CartesianIndices(twins[1])]
        twin_loss(ds, q) = loss(c -> build((x, y), ds[c]), q, _twin_ch, M)
        @testset "$blabel | $klabel | $olabel" begin
            # data side
            g = Zygote.gradient(dd -> (itp = build((x, y), dd); loss(_ -> itp, q0, ch, M)), data)[1]
            for c in 1:M
                ref = ForwardDiff.gradient(dc -> twin_loss(ntuple(k -> k == c ? dc : twins[k], M), q0), twins[c])
                got = g === nothing ? zero(ref) : map(e -> e === nothing ? 0.0 : ch(e, c), g)
                @test got ≈ ref rtol = 1.0e-10 atol = 1.0e-12
            end
            # query side
            if olabel in query_ops
                itp = build((x, y), data)
                gq = Zygote.gradient(qq -> loss(_ -> itp, qq, ch, M), q0)[1]
                refq = ForwardDiff.gradient(v -> twin_loss(ntuple(k -> twins[k], M), Tuple(v)), collect(q0))
                gotq = gq === nothing ? zero(refq) : collect(map(e -> e === nothing ? 0.0 : e, gq))
                @test gotq ≈ refq rtol = 1.0e-10 atol = 1.0e-12
            end
        end
    end
end
