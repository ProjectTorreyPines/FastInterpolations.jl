# ========================================
# ND vector calculus on mixed-unit grids: explicit guards
# ========================================
# `gradient` returns a Tuple, so per-axis components may carry different units
# (`K/s`, `K/m`) and it works. The three APIs that need ONE shared element type
# cannot: `hessian` (Matrix), `laplacian` (a sum), `gradient!` (a user Vector).
# Their component units differ per axis, so the promotion collapses to an
# abstract `Quantity` with no `zero`/`oneunit`.
#
# Without a guard the user meets a raw `DimensionError` from deep inside the
# kernel. These pin an `ArgumentError` naming the API and the workaround, in the
# same style as the hetero-ND unit guard.

@testitem "ND mixed-unit: hessian/laplacian/gradient! refuse with an explanation" begin
    using Unitful

    xs = (0.0:1.0:4.0)u"s"
    ym = (0.0:1.0:3.0)u"m"
    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"
    itp = linear_interp((xs, ym), V)

    @testset "hessian" begin
        e = try
            hessian(itp, 1.5u"s", 1.5u"m")
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        msg = sprint(showerror, e)
        @test occursin("hessian", msg)
        @test occursin("not supported", msg)
        @test occursin("deriv", msg)            # names the per-component workaround
    end

    @testset "laplacian" begin
        e = try
            laplacian(itp, 1.5u"s", 1.5u"m")
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        msg = sprint(showerror, e)
        @test occursin("laplacian", msg)
        @test occursin("not supported", msg)
    end

    @testset "gradient!" begin
        G = Vector{typeof(1.0u"K/s")}(undef, 2)
        e = try
            gradient!(G, itp, 1.5u"s", 1.5u"m")
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        msg = sprint(showerror, e)
        @test occursin("gradient!", msg)
        @test occursin("not supported", msg)
        @test occursin("gradient", msg)         # points at the Tuple-returning form
    end

    # Vector-query and splatted entries route through the same guard.
    @test_throws ArgumentError hessian(itp, [1.5u"s", 1.5u"m"])
    @test_throws ArgumentError laplacian(itp, (1.5u"s", 1.5u"m"))
    @test_throws ArgumentError gradient!(Vector{typeof(1.0u"K/s")}(undef, 2), itp, [1.5u"s", 1.5u"m"])
end

@testitem "ND mixed-unit guard does not fire when the components DO share a type" begin
    using Unitful

    # same-unit axes: every component is `K/s²` → Matrix/sum/Vector all fine
    xs = (0.0:1.0:4.0)u"s"
    ys = (0.0:1.0:3.0)u"s"
    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"
    itp_u = linear_interp((xs, ys), V)
    @test unit(eltype(hessian(itp_u, 1.5u"s", 1.5u"s"))) === u"K" / u"s"^2
    @test unit(laplacian(itp_u, 1.5u"s", 1.5u"s")) === u"K" / u"s"^2
    G = Vector{typeof(1.0u"K/s")}(undef, 2)
    gradient!(G, itp_u, 1.5u"s", 1.5u"s")
    @test unit(eltype(G)) === u"K" / u"s"

    # Real grids of DIFFERENT precision must keep working — they promote to a
    # concrete Float64, unlike units which promote to an abstract Quantity.
    x32 = Float32.(0.0:1.0:4.0)
    y64 = collect(0.0:1.0:3.0)
    Vr = [Float64(i + j) for i in 1:5, j in 1:4]
    itp_r = linear_interp((x32, y64), Vr)
    @test hessian(itp_r, 1.5, 1.5) isa Matrix
    @test laplacian(itp_r, 1.5, 1.5) isa Real
    Gr = zeros(2)
    gradient!(Gr, itp_r, 1.5, 1.5)
    @test Gr isa Vector{Float64}
end
