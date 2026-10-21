# ========================================
# Local-only real-GPU validation: LinearInterpolantND on Metal (Float32)
# ========================================
# Not part of the package test suite (CI has no Apple GPU): real kernel compilation and
# Metal numerics for the ND device leaves and the SoA / AoS batch runner.
#
#   julia --project=test/gpu test/gpu/metal_linear_nd.jl

using Test
using FastInterpolations
using Adapt, GPUArrays, Metal, StaticArrays
using KernelAbstractions: @kernel, @index, @Const, get_backend
const FI = FastInterpolations

include(joinpath(@__DIR__, "linear_nd_cases.jl"))

# A particle pusher's pattern: the interpolant is a kernel argument of the user's own kernel
# and is evaluated point by point (positions as the columns of a 3×n matrix).
@kernel function _field_at!(out, @Const(pos), itp)
    i = @index(Global, Linear)
    @inbounds out[i] = itp((pos[1, i], pos[2, i], pos[3, i]))
end
@kernel function _jacobian_at!(out, @Const(pos), itp)
    i = @index(Global, Linear)
    @inbounds out[i] = jacobian(itp, (pos[1, i], pos[2, i], pos[3, i]))
end
@kernel function _gradient_laplacian_at!(g, l, @Const(pos), itp)
    i = @index(Global, Linear)
    p = @inbounds (pos[1, i], pos[2, i], pos[3, i])
    @inbounds g[i] = SVector(gradient(itp, p))
    @inbounds l[i] = laplacian(itp, p)
end

@testset "Metal: LinearInterpolantND (Float32)" begin
    GPUArrays.allowscalar(false)
    T = Float32
    for (cname, grids) in nd_grid_cases(T)
        N = length(grids)
        data = nd_values(T, grids)
        dmax = maximum(abs, data)
        q_in, q_oob = nd_queries(T, N, 4097)
        itp = linear_interp(grids, data; extrap = ClampExtrap())

        @testset "$cname: kernel argument is isbits and holds device leaves" begin
            k = Metal.mtlconvert(adapt(MtlArray, itp))
            @test isbits(k)
            @test k.data isa FI._DeviceArray{T, N, <:MtlDeviceArray}
        end

        @testset "$cname: parity ($ename)" for (ename, e, qk) in (nd_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            itp_e = linear_interp(grids, data; extrap = e)
            ditp = adapt(MtlArray, itp_e)
            q = qk === :in ? q_in : q_oob
            pts = nd_points(q)
            ref = itp_e(q)
            e isa NoExtrap || @test gpu_close(ditp.(MtlArray(pts)), ref, dmax)
            @test gpu_close(ditp(map(MtlArray, q)), ref, dmax)
            @test gpu_close(ditp(MtlArray(pts)), ref, dmax)
            ops = ntuple(d -> d == 1 ? DerivOp(1) : EvalValue(), N)
            dref = itp_e(q; deriv = ops)
            @test gpu_close(ditp(map(MtlArray, q); deriv = ops), dref, 4 * maximum(abs, filter(!isnan, dref)))
        end

        @testset "$cname: NoExtrap OOB throws the CPU error" begin
            itp_n = linear_interp(grids, data)
            ditp = adapt(MtlArray, itp_n)
            for (qcpu, qgpu) in ((q_oob, map(MtlArray, q_oob)), (nd_points(q_oob), MtlArray(nd_points(q_oob))))
                cpu_err = try
                    itp_n(qcpu)
                catch err
                    err
                end
                gpu_err = try
                    ditp(qgpu)
                catch err
                    err
                end
                @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)
            end
        end

        @testset "$cname: built from device data" begin
            dgrids = map(g -> g isa AbstractRange ? g : MtlArray(collect(g)), grids)
            bitp = linear_interp(dgrids, MtlArray(data); extrap = ClampExtrap())
            @test typeof(bitp) === typeof(adapt(MtlArray, itp))
            @test gpu_close(bitp(map(MtlArray, q_oob)), itp(q_oob), dmax)
        end
    end

    @testset "shaped SoA and AoS queries keep their shape" begin
        grids = (range(0.0f0, 1.0f0; length = 33), nonuniform(T, 17))
        data = nd_values(T, grids)
        itp = linear_interp(grids, data; extrap = ClampExtrap())
        ditp = adapt(MtlArray, itp)
        q = map(v -> reshape(v, 64, 64), nd_queries(T, 2, 4096)[2])
        rm = ditp(map(MtlArray, q))
        @test rm isa MtlArray{T, 2}
        @test gpu_close(rm, itp(q), maximum(abs, data))
        pm = reshape(nd_points(map(vec, q)), 64, 64)
        @test gpu_close(ditp(MtlArray(pm)), itp(pm), maximum(abs, data))
    end

    @testset "evaluated inside a user KernelAbstractions kernel" begin
        grids = (range(0.0f0, 1.0f0; length = 17), nonuniform(T, 13), range(-1.0f0, 1.0f0; length = 9))
        field = linear_interp(grids, [SVector(sin(3a) + c, cos(2b) * a, a * b + c) for a in grids[1], b in grids[2], c in grids[3]]; extrap = ClampExtrap())
        scalar = linear_interp(grids, [sin(3a) * cos(2b) + a * c for a in grids[1], b in grids[2], c in grids[3]]; extrap = ClampExtrap())
        n = 4096
        pos = vcat(T(1.2) .* rand(T, 1, n) .- T(0.1), rand(T, 1, n), 2 .* rand(T, 1, n) .- 1)
        pts = [Tuple(pos[:, k]) for k in 1:n]
        posd = MtlArray(pos)
        be = get_backend(posd)
        for moved in (adapt(MtlArray, field), adapt(be, field))                  # the backend adaptor too
            out = MtlArray{SVector{3, T}}(undef, n)
            _field_at!(be, 256)(out, posd, moved; ndrange = n)
            @test Array(out) ≈ field.(pts)
        end
        jac = MtlArray{SMatrix{3, 3, T, 9}}(undef, n)
        _jacobian_at!(be, 256)(jac, posd, adapt(MtlArray, field); ndrange = n)
        @test Array(jac) ≈ [jacobian(field, p) for p in pts]
        g, l = MtlArray{SVector{3, T}}(undef, n), MtlArray{T}(undef, n)
        _gradient_laplacian_at!(be, 256)(g, l, posd, adapt(MtlArray, scalar); ndrange = n)
        @test Array(g) ≈ [SVector(gradient(scalar, p)) for p in pts]
        @test Array(l) ≈ [laplacian(scalar, p) for p in pts]
    end

    @testset "SVector-valued 3D field, AoS particle positions" begin
        grids = (range(0.0f0, 1.0f0; length = 17), Float32[(i / 16)^1.5 for i in 0:16], range(0.0f0, 2.0f0; length = 9))
        data = [SVector(sin(3a) + c, cos(2b) * a, a * b + c) for a in grids[1], b in grids[2], c in grids[3]]
        itp = linear_interp(grids, data; extrap = ClampExtrap())
        pts = [SVector(T(0.61803398875 * k % 1), T(0.41421356237 * k % 1), 2 * T(0.73205080757 * k % 1)) for k in 1:4097]
        ref = itp.(Tuple.(pts))
        res = Array(adapt(MtlArray, itp)(MtlArray(pts)))
        @test maximum(i -> maximum(abs, res[i] - ref[i]), eachindex(ref)) <= 16 * eps(T)
    end
end
