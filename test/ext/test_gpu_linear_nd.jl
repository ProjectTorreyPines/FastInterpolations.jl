# ========================================
# GPU storage and evaluation for LinearInterpolantND (JLArrays)
# ========================================
# Same contracts as the 1D file (device leaves, Adapt rebuilds, batch runner, GPU-native
# construction) for tensor-product linear ND, with SoA (tuple of vectors) and AoS (vector of
# points) GPU queries. Real kernel compilation: test/gpu/metal_linear_nd.jl.

using Test
using FastInterpolations
using Adapt, GPUArrays, JLArrays, StaticArrays
const FI = FastInterpolations

include(joinpath(@__DIR__, "..", "gpu", "linear_nd_cases.jl"))

# Host precision-changing adaptor, as `Metal.mtl` / `CUDA.cu` do on their way to the device.
struct _NDFloat32Adaptor end
Adapt.adapt_storage(::_NDFloat32Adaptor, x::Array{Float64}) = Float32.(x)

@testset "GPU storage: LinearInterpolantND (JLArrays)" begin
    GPUArrays.allowscalar(false)
    @test fieldnames(FI.LinearInterpolantND) == (:grids, :data, :extraps, :searches)

    for T in (Float32, Float64), (cname, grids) in nd_grid_cases(T)
        N = length(grids)
        data = nd_values(T, grids)
        dmax = maximum(abs, data)
        q_in, q_oob = nd_queries(T, N, 211)
        itp = linear_interp(grids, data; extrap = ClampExtrap())

        @testset "$T $cname: device form and round trip" begin
            ditp = adapt(JLArray, itp)
            @test ditp.data isa FI._DeviceArray{T, N}
            @test parent(ditp.data) isa JLArray
            for (g, gd) in zip(itp.grids, ditp.grids)
                @test g isa FI._CachedRange ? gd === g : gd isa FI._DeviceVector{T}
            end
            @test adapt(Array, itp) === itp
            back = adapt(Array, ditp)
            @test typeof(back) === typeof(itp)
            k = JLArrays.jlconvert(ditp)
            @test k.data isa FI._DeviceArray && !(parent(k.data) isa JLArray)
            @test occursin("LinearInterpolantND", sprint(show, MIME"text/plain"(), ditp))
        end

        @testset "$T $cname: broadcast and batch parity ($ename)" for (ename, e, qk) in (nd_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            itp_e = linear_interp(grids, data; extrap = e)
            ditp = adapt(JLArray, itp_e)
            q = qk === :in ? q_in : q_oob
            pts = nd_points(q)
            ref = itp_e(q)
            e isa NoExtrap || @test gpu_close(ditp.(JLArray(pts)), ref, dmax)       # scalar path per element
            @test gpu_close(ditp(map(JLArray, q)), ref, dmax)                        # SoA batch
            @test gpu_close(ditp(JLArray(pts)), ref, dmax)                           # AoS batch
            out = JLArray(similar(ref))
            @test ditp(out, map(JLArray, q)) === out
            @test gpu_close(out, ref, dmax)
            # Derivatives of extended corners amplify a few-ulp codegen difference (muladd
            # contraction in the kernel vs the host loop): a 4× looser magnitude scale.
            ops = ntuple(d -> d == 1 ? DerivOp(1) : EvalValue(), N)
            dref = itp_e(q; deriv = ops)
            @test gpu_close(ditp(map(JLArray, q); deriv = ops), dref, 4 * maximum(abs, filter(!isnan, dref)))
        end

        @testset "$T $cname: batch planning" begin
            itp_n = linear_interp(grids, data)                                       # NoExtrap
            ditp = adapt(JLArray, itp_n)
            for (qcpu, qgpu) in ((q_oob, map(JLArray, q_oob)), (nd_points(q_oob), JLArray(nd_points(q_oob))))
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
                @test gpu_err isa DomainError
                @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)      # same point, axis, bounds
            end
            @test gpu_close(ditp(map(JLArray, q_in); extrap = InBounds()), itp_n(q_in), dmax)
            @test_throws ArgumentError ditp(map(JLArray, q_in); hint = ntuple(_ -> Ref(1), N))
            @test length(ditp(map(_ -> JLArray(T[]), q_in))) == 0
        end

        @testset "$T $cname: built from device data" begin
            dgrids = map(g -> g isa AbstractRange ? g : JLArray(collect(g)), grids)
            bitp = linear_interp(dgrids, JLArray(data); extrap = ClampExtrap())
            @test typeof(bitp) === typeof(adapt(JLArray, itp))
            @test gpu_close(bitp(map(JLArray, q_oob)), itp(q_oob), dmax)
            ritp = linear_interp(dgrids, JLArray(data); extrap = ClampExtrap(), store = StorePolicy(copy = false))
            @test ritp.data isa FI._DeviceArray{T, N}
        end
    end

    @testset "SVector-valued 3D field (AoS particles)" begin
        grids = (range(0.0f0, 1.0f0; length = 17), Float32[(i / 16)^1.5 for i in 0:16], range(0.0f0, 2.0f0; length = 9))
        data = [SVector(sin(3a) + c, cos(2b) * a, a * b + c) for a in grids[1], b in grids[2], c in grids[3]]
        itp = linear_interp(grids, data; extrap = ClampExtrap())
        ditp = adapt(JLArray, itp)
        @test ditp.data isa FI._DeviceArray{SVector{3, Float32}, 3}
        pts = [SVector(rand(Float32), rand(Float32), 2 * rand(Float32)) for _ in 1:97]
        ref = itp.(Tuple.(pts))
        res = Array(ditp(JLArray(pts)))
        @test maximum(i -> maximum(abs, res[i] - ref[i]), eachindex(ref)) <= 4 * eps(Float32) * 4
    end

    # A precision-changing move narrows every grid with the data and keeps each range tag
    # (`_WidenedDomain` built directly so that every architecture runs it).
    @testset "precision-changing adaptor (mixed range tags)" begin
        wid = FI._cached_range(FI._WidenedDomain(), 0.0, 1.0, 1 / 8, 8.0, 9)
        grids = (1:9, Base.OneTo(7), wid)
        data = [sin(a / 3) + cos(b / 2) * c for a in 1:9, b in 1:7, c in collect(wid)]
        itp = linear_interp(grids, data; extrap = ClampExtrap())
        itp32 = adapt(_NDFloat32Adaptor(), itp)
        @test eltype(itp32.data) === Float32
        @test itp32.grids[1] isa FI._CachedRange{Float32, Float32, FI._UnitStep}
        @test itp32.grids[2] isa FI._CachedRange{Float32, Float32, FI._OneTo}
        @test itp32.grids[3] isa FI._CachedRange{Float32, Float32, FI._WidenedDomain}
        ref = linear_interp(map(g -> Float32.(collect(g)), grids), Float32.(data); extrap = ClampExtrap())
        q = (Float32[0.5, 4.2, 9.5], Float32[1.0, 3.3, 7.5], Float32[-0.1, 0.4, 1.2])
        @test gpu_close(itp32(q), ref(q), 2.0f0)
    end

    @testset "ND batch admission" begin
        grids = (range(0.0f0, 1.0f0; length = 9), range(0.0f0, 1.0f0; length = 7))
        data = rand(Float32, 9, 7)
        q = (JLArray(rand(Float32, 8)), JLArray(rand(Float32, 8)))
        @test_throws ArgumentError linear_interp(grids, data)(q)                   # data on the host
        @test_throws ArgumentError cubic_interp(grids, data)(q)                    # family not admitted
    end
end
