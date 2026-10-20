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
const FI = FastInterpolations

include(joinpath(@__DIR__, "linear_nd_cases.jl"))

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
