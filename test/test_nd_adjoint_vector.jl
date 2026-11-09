# ========================================
# ND adjoints with vector-valued cotangents
# ========================================
# The adjoint is linear in the cotangent, so a vector-valued (or complex) seed must
# equal the Float64 adjoint applied channel by channel — for every family and BC.

@testitem "ND adjoint: vector-valued seeds match the per-channel Float64 adjoint" begin
    using StaticArrays

    x = range(0.0, 2.0, 11)
    y = range(0.0, 1.0, 9)
    q = (0.7, 0.4)
    d = [sin(π * a) * cos(b) for a in x, b in y]
    d[end, :] .= d[1, :]                                  # exactly periodic in x

    configs = (
        ("linear", linear_interp((x, y), d)),
        ("constant", constant_interp((x, y), d)),
        ("quadratic", quadratic_interp((x, y), d)),
        ("cubic", cubic_interp((x, y), d)),
        ("cubic PeriodicBC", cubic_interp((x, y), d; bc = (PeriodicBC(), CubicFit()))),
        (
            "cubic PeriodicBC exclusive",
            cubic_interp((x[1:(end - 1)], y), d[1:(end - 1), :]; bc = (PeriodicBC(endpoint = :exclusive, period = 2.0), CubicFit())),
        ),
        ("hetero cubic × linear", interp((x, y), d; method = (CubicInterp(), LinearInterp()))),
    )
    # (label, channels → seed, seed → channels)
    kinds = (
        ("SVector{3}", c -> SVector(c...), s -> Tuple(s)),
        ("ComplexF64", c -> complex(c[1], c[2]), s -> (real(s), imag(s), 0.0)),
    )
    chans = (0.7, -1.3, 2.1)
    all_ops = (
        (EvalValue(), EvalValue()), (DerivOp(1), EvalValue()),
        (EvalValue(), DerivOp(1)), (DerivOp(1), DerivOp(1)),
    )

    for (clabel, itp) in configs
        adj = FastInterpolations._adjoint_func_from_itp(itp)(
            itp.grids, (q,); FastInterpolations._adjoint_kwargs_from_itp(itp)...
        )
        for ops in all_ops
            refs = map(c -> adj([c]; deriv = ops), chans)
            for (klabel, mk, unpack) in kinds
                M = klabel == "ComplexF64" ? 2 : 3
                @testset "$clabel | $ops | $klabel" begin
                    r = adj([mk(chans)]; deriv = ops)
                    @test all(k -> all(i -> isapprox(unpack(r[i])[k], refs[k][i]; atol = 1.0e-12), eachindex(r)), 1:M)
                end
            end
        end
    end
end
