# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║          ND jacobian / value_jacobian — Zygote reverse mode (scalar)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# `jacobian` is `hcat(gradient(itp, q)...)`, so its reverse pass runs through the
# `gradient` / `value_gradient` rrules. `hcat`'s rule hands each component back as a
# lazy thunk, which the pullbacks must unthunk one by one. Oracle: ForwardDiff on the
# same interpolant (data side and query side).

using Test
using FastInterpolations
using Zygote
using ForwardDiff

# Zygote returns `nothing` for an all-zero gradient (constant interpolants).
_or_zero(g, ref) = g === nothing ? zero(ref) : g
_or_zero(g::Tuple, _) = collect(map(gi -> gi === nothing ? 0.0 : gi, g))

@testset "ND jacobian — Zygote (scalar data)" begin
    x = range(0.0, 2.0, 11)
    y = range(0.0, 1.0, 9)
    z = range(0.0, 1.0, 6)
    data2 = [sin(a) * cos(b) + 0.3 * a * b for a in x, b in y]
    data3 = [sin(a) * cos(b) * exp(-c) + 0.2 * a * c for a in x, b in y, c in z]
    q2 = (0.7, 0.4)
    q3 = (0.7, 0.4, 0.55)

    builders2 = (
        ("linear", linear_interp),
        ("quadratic", quadratic_interp),
        ("constant", constant_interp),
        ("cubic", cubic_interp),
        ("hetero", (g, d) -> interp(g, d; method = (CubicInterp(), LinearInterp()))),
    )
    W = [1.3 -0.7]
    L(J) = sum(W .* J .^ 2) + J[1, 2]          # both columns, nonlinear
    L2(J) = J[1, 2]^2                            # second column only

    @testset "$label 2D" for (label, build) in builders2
        itp = build((x, y), data2)

        @testset "jacobian ∂data" begin
            for loss in (L, L2)
                g = Zygote.gradient(d -> loss(FastInterpolations.jacobian(build((x, y), d), q2)), data2)[1]
                ref = ForwardDiff.gradient(d -> loss(FastInterpolations.jacobian(build((x, y), d), q2)), data2)
                @test _or_zero(g, ref) ≈ ref rtol = 1.0e-10 atol = 1.0e-12
            end
        end

        @testset "jacobian ∂query" begin
            for loss in (L, L2)
                g = Zygote.gradient(q -> loss(FastInterpolations.jacobian(itp, q)), q2)[1]
                ref = ForwardDiff.gradient(v -> loss(FastInterpolations.jacobian(itp, v)), collect(q2))
                @test _or_zero(g, ref) ≈ ref rtol = 1.0e-10 atol = 1.0e-12
            end
        end

        @testset "value_jacobian ∂data / ∂query" begin
            Lv(vJ) = vJ[1]^2 + L(vJ[2])
            g = Zygote.gradient(d -> Lv(FastInterpolations.value_jacobian(build((x, y), d), q2)), data2)[1]
            ref = ForwardDiff.gradient(d -> Lv(FastInterpolations.value_jacobian(build((x, y), d), q2)), data2)
            @test _or_zero(g, ref) ≈ ref rtol = 1.0e-10 atol = 1.0e-12

            gq = Zygote.gradient(q -> Lv(FastInterpolations.value_jacobian(itp, q)), q2)[1]
            refq = ForwardDiff.gradient(v -> Lv(FastInterpolations.value_jacobian(itp, v)), collect(q2))
            @test _or_zero(gq, refq) ≈ refq rtol = 1.0e-10 atol = 1.0e-12
        end
    end

    @testset "query forms agree (cubic 2D)" begin
        f_tuple(d) = L(FastInterpolations.jacobian(cubic_interp((x, y), d), q2))
        f_splat(d) = L(FastInterpolations.jacobian(cubic_interp((x, y), d), q2...))
        f_vec(d) = L(FastInterpolations.jacobian(cubic_interp((x, y), d), collect(q2)))
        g_tuple = Zygote.gradient(f_tuple, data2)[1]
        @test Zygote.gradient(f_splat, data2)[1] ≈ g_tuple rtol = 1.0e-12
        @test Zygote.gradient(f_vec, data2)[1] ≈ g_tuple rtol = 1.0e-12

        itp = cubic_interp((x, y), data2)
        ref = ForwardDiff.gradient(v -> L(FastInterpolations.jacobian(itp, v)), collect(q2))
        @test collect(Zygote.gradient(v -> L(FastInterpolations.jacobian(itp, v)), collect(q2))[1]) ≈ ref rtol = 1.0e-10
    end

    @testset "$label 3D" for (label, build) in (("linear", linear_interp), ("cubic", cubic_interp))
        L3(J) = sum(abs2, J) + J[1, 3]
        g = Zygote.gradient(d -> L3(FastInterpolations.jacobian(build((x, y, z), d), q3)), data3)[1]
        ref = ForwardDiff.gradient(d -> L3(FastInterpolations.jacobian(build((x, y, z), d), q3)), data3)
        @test g ≈ ref rtol = 1.0e-10 atol = 1.0e-12

        itp = build((x, y, z), data3)
        gq = Zygote.gradient(q -> L3(FastInterpolations.jacobian(itp, q)), q3)[1]
        refq = ForwardDiff.gradient(v -> L3(FastInterpolations.jacobian(itp, v)), collect(q3))
        @test collect(gq) ≈ refq rtol = 1.0e-10 atol = 1.0e-12
    end

    # Root cause, independent of `jacobian`: a gradient tuple fed through `hcat`.
    @testset "gradient / value_gradient through hcat (thunked components)" begin
        f(d) = sum(abs2, hcat(FastInterpolations.gradient(cubic_interp((x, y), d), q2)...))
        @test Zygote.gradient(f, data2)[1] ≈ ForwardDiff.gradient(f, data2) rtol = 1.0e-10

        fv(d) = (vg = FastInterpolations.value_gradient(cubic_interp((x, y), d), q2); vg[1] + sum(abs2, hcat(vg[2]...)))
        @test Zygote.gradient(fv, data2)[1] ≈ ForwardDiff.gradient(fv, data2) rtol = 1.0e-10
    end
end
