# ========================================
# Real-GPU tests: LinearInterpolantND (included by test/gpu/runtests.jl)
# ========================================
# Real kernel compilation and device numerics for the ND device leaves, SoA / AoS batch
# calls and evaluation inside user KernelAbstractions kernels. Backend-agnostic, as the 1D file.

using Test
using FastInterpolations
using Adapt, GPUArrays, StaticArrays
using KernelAbstractions: @kernel, @index, @Const

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

@testset "$GPU_NAME: LinearInterpolantND ($T)" for T in GPU_ELTYPES
    GPUArrays.allowscalar(false)
    for (cname, grids) in nd_grid_cases(T)
        N = length(grids)
        data = nd_values(T, grids)
        dmax = maximum(abs, data)
        q_in, q_oob = nd_queries(T, N, 4097)
        itp = linear_interp(grids, data; extrap = ClampExtrap())

        @testset "$cname: kernel argument is isbits and holds device leaves" begin
            k = kernel_arg(adapt(GPUArrayType, itp))
            @test isbits(k)
            @test k.data isa FI._DeviceArray{T, N} && isbits(parent(k.data))
        end

        @testset "$cname: parity ($ename)" for (ename, e, qk) in (nd_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            itp_e = linear_interp(grids, data; extrap = e)
            ditp = adapt(GPUArrayType, itp_e)
            q = qk === :in ? q_in : q_oob
            pts = nd_points(q)
            ref = itp_e(q)
            e isa NoExtrap || @test gpu_close(ditp.(dev(pts)), ref, dmax)
            @test gpu_close(ditp(map(dev, q)), ref, dmax)
            @test gpu_close(ditp(dev(pts)), ref, dmax)
            ops = ntuple(d -> d == 1 ? DerivOp(1) : EvalValue(), N)
            dref = itp_e(q; deriv = ops)
            @test gpu_close(ditp(map(dev, q); deriv = ops), dref, 4 * maximum(abs, filter(!isnan, dref)))
        end

        @testset "$cname: NoExtrap OOB throws the CPU error" begin
            itp_n = linear_interp(grids, data)
            ditp = adapt(GPUArrayType, itp_n)
            for (qcpu, qgpu) in ((q_oob, map(dev, q_oob)), (nd_points(q_oob), dev(nd_points(q_oob))))
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
            dgrids = map(g -> g isa AbstractRange ? g : dev(collect(g)), grids)
            bitp = linear_interp(dgrids, dev(data); extrap = ClampExtrap())
            @test typeof(bitp) === typeof(adapt(GPUArrayType, itp))
            @test gpu_close(bitp(map(dev, q_oob)), itp(q_oob), dmax)
        end
    end

    @testset "shaped SoA and AoS queries keep their shape" begin
        grids = (range(zero(T), one(T); length = 33), nonuniform(T, 17))
        data = nd_values(T, grids)
        itp = linear_interp(grids, data; extrap = ClampExtrap())
        ditp = adapt(GPUArrayType, itp)
        q = map(v -> reshape(v, 64, 64), nd_queries(T, 2, 4096)[2])
        rm = ditp(map(dev, q))
        @test rm isa AbstractArray{T, 2} && on_device(rm)
        @test gpu_close(rm, itp(q), maximum(abs, data))
        pm = reshape(nd_points(map(vec, q)), 64, 64)
        @test gpu_close(ditp(dev(pm)), itp(pm), maximum(abs, data))
    end

    @testset "evaluated inside a user KernelAbstractions kernel" begin
        grids = (range(zero(T), one(T); length = 17), nonuniform(T, 13), range(-one(T), one(T); length = 9))
        fv = [SVector(sin(3a) + c, cos(2b) * a, a * b + c) for a in grids[1], b in grids[2], c in grids[3]]
        field = linear_interp(grids, fv; extrap = ClampExtrap())
        scalar = linear_interp(grids, [sin(3a) * cos(2b) + a * c for a in grids[1], b in grids[2], c in grids[3]]; extrap = ClampExtrap())
        n = 4096
        pos = vcat(T(1.2) .* rand(T, 1, n) .- T(0.1), rand(T, 1, n), 2 .* rand(T, 1, n) .- 1)   # x partly outside
        pts = [Tuple(pos[:, k]) for k in 1:n]
        posd = dev(pos)
        for moved in (adapt(GPUArrayType, field), adapt(BACKEND, field))       # the array type and the backend
            out = KernelAbstractions.allocate(BACKEND, SVector{3, T}, n)
            _field_at!(BACKEND, 256)(out, posd, moved; ndrange = n)
            @test Array(out) ≈ field.(pts)
        end
        # Outside the box the field is a fill value (a particle leaving the domain sees zero).
        zfield = linear_interp(grids, fv; extrap = FillExtrap(zero(SVector{3, T})))
        out = KernelAbstractions.allocate(BACKEND, SVector{3, T}, n)
        _field_at!(BACKEND, 256)(out, posd, adapt(GPUArrayType, zfield); ndrange = n)
        @test Array(out) ≈ zfield.(pts)
        @test count(iszero, Array(out)) == count(iszero, zfield.(pts)) > 0
        jac = KernelAbstractions.allocate(BACKEND, SMatrix{3, 3, T, 9}, n)
        _jacobian_at!(BACKEND, 256)(jac, posd, adapt(GPUArrayType, field); ndrange = n)
        @test Array(jac) ≈ [jacobian(field, p) for p in pts]
        g, l = KernelAbstractions.allocate(BACKEND, SVector{3, T}, n), KernelAbstractions.allocate(BACKEND, T, n)
        _gradient_laplacian_at!(BACKEND, 256)(g, l, posd, adapt(GPUArrayType, scalar); ndrange = n)
        @test Array(g) ≈ [SVector(gradient(scalar, p)) for p in pts]
        @test Array(l) ≈ [laplacian(scalar, p) for p in pts]
    end

    @testset "SVector-valued 3D field, AoS particle positions" begin
        grids = (range(zero(T), one(T); length = 17), T[(i / 16)^1.5 for i in 0:16], range(zero(T), T(2); length = 9))
        data = [SVector(sin(3a) + c, cos(2b) * a, a * b + c) for a in grids[1], b in grids[2], c in grids[3]]
        itp = linear_interp(grids, data; extrap = ClampExtrap())
        pts = [SVector(T(0.61803398875 * k % 1), T(0.41421356237 * k % 1), 2 * T(0.73205080757 * k % 1)) for k in 1:4097]
        ref = itp.(Tuple.(pts))
        res = Array(adapt(GPUArrayType, itp)(dev(pts)))
        @test maximum(i -> maximum(abs, res[i] - ref[i]), eachindex(ref)) <= 16 * eps(T)
    end
end
