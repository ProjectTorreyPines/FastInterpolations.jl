# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║     All Methods × Series × Dual Grid — Comprehensive Coverage             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Cross-cutting test: ensures every method's Series oneshot path works with
# Dual grids across all combinations:
#   Grid type: Vector{Dual}, Range{Dual} (collected to Vector)
#   Query mode: scalar, vector, in-place scalar
#
# NOTE: PCHIP/Cardinal/Akima do not have Series API — excluded.

using Test
using FastInterpolations
using ForwardDiff

@testset "Series × Dual Grid — all methods" begin

    x_vec = collect(range(0.0, 5.0, 20))
    x_range = range(0.0, 5.0, 20)
    y1 = sin.(x_vec)
    y2 = cos.(x_vec)
    d = ForwardDiff.Dual{:tag}(1.0, 1.0)
    xd_vec = d .* x_vec
    xd_range_vec = collect(d .* x_range)  # Range{Dual} → Vector{Dual}
    xq = 2.5
    xq_vec = [1.0, 2.5, 4.0]
    s = Series(y1, y2)

    # Helper: check primals match Float reference
    check_primals(dual_vals, float_vals) = ForwardDiff.value.(dual_vals) ≈ float_vals

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CONSTANT                                                              ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "constant" begin
        ref_s = constant_interp(x_vec, s, xq; extrap = ExtendExtrap())
        ref_v = constant_interp(x_vec, s, xq_vec; extrap = ExtendExtrap())

        @testset "Vector grid — scalar query" begin
            vals = constant_interp(xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(vals, ref_s)
        end
        @testset "Vector grid — vector query" begin
            vals = constant_interp(xd_vec, s, xq_vec; extrap = ExtendExtrap())
            @test check_primals(vals[1], ref_v[1])
        end
        @testset "Vector grid — in-place scalar" begin
            output = Vector{Any}(undef, 2)
            constant_interp!(output, xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(output, ref_s)
        end
        @testset "Range grid — scalar query" begin
            # Range broadcasting (d .* range) has ULP differences vs d .* collect(range).
            # Constant is discontinuous (nearest-neighbor) so ULP boundary shift may pick
            # adjacent y-value. Check result is a valid y-value (within atol of some y[k]).
            vals = constant_interp(xd_range_vec, s, xq; extrap = ExtendExtrap())
            @test any(v -> isapprox(ForwardDiff.value(vals[1]), v; atol = 1.0e-14), y1)
            @test any(v -> isapprox(ForwardDiff.value(vals[2]), v; atol = 1.0e-14), y2)
        end
        @testset "Range grid — vector query" begin
            vals = constant_interp(xd_range_vec, s, xq_vec; extrap = ExtendExtrap())
            # Each result element must be a valid y-value
            @test all(r -> any(v -> isapprox(ForwardDiff.value(r), v; atol = 1.0e-14), y1), vals[1])
            @test all(r -> any(v -> isapprox(ForwardDiff.value(r), v; atol = 1.0e-14), y2), vals[2])
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  LINEAR                                                                ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "linear" begin
        ref_s = linear_interp(x_vec, s, xq; extrap = ExtendExtrap())
        ref_v = linear_interp(x_vec, s, xq_vec; extrap = ExtendExtrap())

        @testset "Vector grid — scalar query" begin
            vals = linear_interp(xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(vals, ref_s)
        end
        @testset "Vector grid — vector query" begin
            vals = linear_interp(xd_vec, s, xq_vec; extrap = ExtendExtrap())
            @test check_primals(vals[1], ref_v[1])
        end
        @testset "Vector grid — in-place scalar" begin
            output = Vector{Any}(undef, 2)
            linear_interp!(output, xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(output, ref_s)
        end
        @testset "Range grid — scalar query" begin
            vals = linear_interp(xd_range_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(vals, ref_s)
        end
        @testset "Range grid — vector query" begin
            vals = linear_interp(xd_range_vec, s, xq_vec; extrap = ExtendExtrap())
            @test check_primals(vals[1], ref_v[1])
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  QUADRATIC                                                             ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "quadratic" begin
        ref_s = quadratic_interp(x_vec, s, xq; extrap = ExtendExtrap())
        ref_v = quadratic_interp(x_vec, s, xq_vec; extrap = ExtendExtrap())

        @testset "Vector grid — scalar query" begin
            vals = quadratic_interp(xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(vals, ref_s)
        end
        @testset "Vector grid — vector query" begin
            vals = quadratic_interp(xd_vec, s, xq_vec; extrap = ExtendExtrap())
            @test check_primals(vals[1], ref_v[1])
        end
        @testset "Vector grid — in-place scalar" begin
            output = Vector{Any}(undef, 2)
            quadratic_interp!(output, xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(output, ref_s)
        end
        @testset "Range grid — scalar query" begin
            vals = quadratic_interp(xd_range_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(vals, ref_s)
        end
        @testset "Range grid — vector query" begin
            vals = quadratic_interp(xd_range_vec, s, xq_vec; extrap = ExtendExtrap())
            @test check_primals(vals[1], ref_v[1])
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CUBIC                                                                 ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "cubic" begin
        ref_s = cubic_interp(x_vec, s, xq; extrap = ExtendExtrap())
        ref_v = cubic_interp(x_vec, s, xq_vec; extrap = ExtendExtrap())

        @testset "Vector grid — scalar query" begin
            vals = cubic_interp(xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(vals, ref_s)
        end
        @testset "Vector grid — vector query" begin
            vals = cubic_interp(xd_vec, s, xq_vec; extrap = ExtendExtrap())
            @test check_primals(vals[1], ref_v[1])
        end
        @testset "Vector grid — in-place scalar" begin
            Tout = ForwardDiff.Dual{:tag, Float64, 1}
            output = Vector{Tout}(undef, 2)
            cubic_interp!(output, xd_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(output, ref_s)
        end
        @testset "Range grid — scalar query" begin
            vals = cubic_interp(xd_range_vec, s, xq; extrap = ExtendExtrap())
            @test check_primals(vals, ref_s)
        end
        @testset "Range grid — vector query" begin
            vals = cubic_interp(xd_range_vec, s, xq_vec; extrap = ExtendExtrap())
            @test check_primals(vals[1], ref_v[1])
        end
    end

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  SERIES INTERPOLANT (2-arg, persistent) with Dual grid                 ║
    # ║  Tests LazyTranspose/Pair/Triple with Tz ≠ Tv                          ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @testset "constant Series interpolant — Dual grid" begin
        sitp = constant_interp(xd_vec, s; extrap = ExtendExtrap())
        vals = sitp(xq)
        sitp_f = constant_interp(x_vec, s; extrap = ExtendExtrap())
        @test check_primals(vals, sitp_f(xq))
    end

    @testset "linear Series interpolant — Dual grid" begin
        sitp = linear_interp(xd_vec, s; extrap = ExtendExtrap())
        vals = sitp(xq)
        sitp_f = linear_interp(x_vec, s; extrap = ExtendExtrap())
        @test check_primals(vals, sitp_f(xq))
    end

    @testset "quadratic Series interpolant — Dual grid" begin
        sitp = quadratic_interp(xd_vec, s; extrap = ExtendExtrap())
        vals = sitp(xq)
        sitp_f = quadratic_interp(x_vec, s; extrap = ExtendExtrap())
        @test check_primals(vals, sitp_f(xq))
        @test any(!iszero, ForwardDiff.partials.(vals))
    end

    @testset "cubic Series interpolant — Dual grid" begin
        sitp = cubic_interp(xd_vec, s; extrap = ExtendExtrap())
        vals = sitp(xq)
        sitp_f = cubic_interp(x_vec, s; extrap = ExtendExtrap())
        @test check_primals(vals, sitp_f(xq))

        # Verify partials are nonzero — grid sensitivity actually propagates
        @test any(!iszero, ForwardDiff.partials.(vals))
    end

end
