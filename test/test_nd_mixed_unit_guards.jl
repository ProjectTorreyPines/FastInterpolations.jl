# ========================================
# ND vector calculus on mixed-unit grids: values where they exist, guards where not
# ========================================
# `gradient` (Tuple) and `hessian` (Matrix with the components' promoted —
# possibly abstract — eltype) return VALUES: every component exists, only a
# shared concrete type may not. `laplacian` stays guarded (its terms must be
# ADDED across axes — dimensionally undefined for mixed units), and the
# in-place forms refuse only a store that genuinely cannot hold the components.

@testitem "ND mixed-unit: hessian returns the abstract-eltype matrix; laplacian/gradient! guard" begin
    using Unitful

    xs = (0.0:1.0:4.0)u"s"
    ym = (0.0:1.0:3.0)u"m"
    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"
    itp = linear_interp((xs, ym), V)

    @testset "hessian: per-element units, values ≡ component deriv queries" begin
        H = hessian(itp, 1.5u"s", 1.5u"m")
        @test !isconcretetype(eltype(H))
        @test eltype(H) <: Quantity{Float64}
        @test unit(H[1, 1]) === u"K" / u"s"^2
        @test unit(H[2, 2]) === u"K" / u"m"^2
        @test H[1, 2] === itp((1.5u"s", 1.5u"m"); deriv = (DerivOp(1), DerivOp(1)))
        @test H[2, 1] === H[1, 2]
        # Vector-query form agrees.
        Hv = hessian(itp, [1.5u"s", 1.5u"m"])
        @test all(i -> Hv[i] === H[i], eachindex(H))
    end

    @testset "hessian: FillExtrap OOB zeros carry per-element units" begin
        itp_f = linear_interp((xs, ym), V; extrap = FillExtrap(0.0u"K"))
        Ho = hessian(itp_f, 9.0u"s", 1.5u"m")
        @test unit(Ho[1, 1]) === u"K" / u"s"^2
        @test unit(Ho[1, 2]) === u"K" / (u"s" * u"m")
        @test all(iszero, Ho)
        # In-place with an Any store takes the same per-element OOB path
        # (was: fill!(H, zero(Any)) → MethodError).
        Ha = Matrix{Any}(undef, 2, 2)
        hessian!(Ha, itp_f, 9.0u"s", 1.5u"m")
        @test unit(Ha[2, 2]) === u"K" / u"m"^2
        @test all(iszero, Ha)
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

    # Vector-query and splatted entries route through the same guards.
    @test_throws ArgumentError laplacian(itp, (1.5u"s", 1.5u"m"))
    @test_throws ArgumentError gradient!(Vector{typeof(1.0u"K/s")}(undef, 2), itp, [1.5u"s", 1.5u"m"])
end

@testitem "ND mixed-unit: an in-place store that CAN hold the components is accepted" begin
    using Unitful

    # The in-place guards exist because the caller's buffer needs one element
    # type. Whether it has one is a property of the BUFFER, not of the grid: an
    # `Any` (or abstract-`Quantity`) store holds `K/s` beside `K/m` perfectly
    # well, and the generic kernels already fill it correctly. Rejecting it makes
    # the guard stricter than the thing it guards.
    xs = (0.0:1.0:4.0)u"s"
    ym = (0.0:1.0:3.0)u"m"
    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"
    itp = linear_interp((xs, ym), V)

    G = Vector{Any}(undef, 2)
    gradient!(G, itp, 1.5u"s", 1.5u"m")
    @test G[1] ≈ gradient(itp, 1.5u"s", 1.5u"m")[1]
    @test G[2] ≈ gradient(itp, 1.5u"s", 1.5u"m")[2]

    Gq = Vector{Quantity{Float64}}(undef, 2)
    gradient!(Gq, itp, 1.5u"s", 1.5u"m")
    @test unit(Gq[1]) === u"K" / u"s"
    @test unit(Gq[2]) === u"K" / u"m"

    H = Matrix{Any}(undef, 2, 2)
    hessian!(H, itp, 1.5u"s", 1.5u"m")
    @test unit(H[1, 1]) === u"K" / u"s"^2
    @test unit(H[2, 2]) === u"K" / u"m"^2
    @test unit(H[1, 2]) === unit(H[2, 1])

    # A buffer that genuinely cannot hold them is still refused, with the same
    # explanatory error (pinned in the testitem above).
    @test_throws ArgumentError gradient!(Vector{Float64}(undef, 2), itp, 1.5u"s", 1.5u"m")
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
