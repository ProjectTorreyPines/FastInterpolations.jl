# ========================================
# Local-only real-GPU validation: 1D LinearInterpolant on Metal (Float32)
# ========================================
# Not part of the package test suite (no @testitem; CI has no Apple GPU). Covers what the
# JLArrays extension test cannot: real kernel compilation and Metal numerics.
#
#   julia --project=test/gpu test/gpu/metal_linear1d.jl

using Test
using FastInterpolations
using Adapt, GPUArrays, Metal
const FI = FastInterpolations

include(joinpath(@__DIR__, "linear1d_cases.jl"))

is_invalid_ir(err) = nameof(typeof(err)) === :InvalidIRError

# Minimum summed device time of the command buffers committed by `f`.
function device_time(f; reps = 20)
    f()
    Metal.synchronize()
    return minimum(1:reps) do _
        d = Metal.Profiling.profile_internally(f).device
        sum(d.stop .- d.start)
    end
end

@testset "Metal: LinearInterpolant 1D (Float32)" begin
    GPUArrays.allowscalar(false)
    T = Float32
    @test Metal.functional()

    @testset "extensions load; no backend-specific extension" begin
        @test Base.get_extension(FastInterpolations, :FastInterpolationsAdaptExt) !== nothing
        @test Base.get_extension(FastInterpolations, :FastInterpolationsGPUArraysExt) !== nothing
        @test Base.get_extension(FastInterpolations, :FastInterpolationsMetalExt) === nothing
    end

    @testset "batch admission: host data and unsupported families are rejected" begin
        x = range(zero(T), one(T); length = 33)
        q = MtlArray(gpu_queries(T, 64)[1])
        @test_throws ArgumentError linear_interp(x, gpu_fy.(x))(q)       # not on the device yet
        @test_throws ArgumentError cubic_interp(x, gpu_fy.(x))(q)        # family not admitted
    end

    @testset "persistent shaped queries keep their shape ($gname)" for (gname, x) in gpu_grids(T, 33)
        itp = linear_interp(x, gpu_fy.(x); extrap = ClampExtrap())
        ditp = adapt(MtlArray, itp)
        qm = reshape(gpu_queries(T, 4096)[2], 64, 64)
        rm = ditp(MtlArray(qm))
        @test rm isa MtlArray{T, 2}
        @test gpu_close(rm, itp(qm), 2.0f0)
        dm = itp(qm; deriv = DerivOp(1))
        @test gpu_close(ditp(MtlArray(qm); deriv = DerivOp(1)), dm, maximum(abs, dm))
    end

    # Uniform grids: the isbits `_CachedRange` stays a range inside Metal kernels, with
    # Float32-only fields even for Base `range` input (Float64 internals) and Int `1:n`.
    @testset "range grid ($rname) stays a range on the device" for (rname, x) in gpu_range_grids(T, 65)
        y = gpu_values(T, x)
        ymax = maximum(abs, y)
        q_in, q_oob = gpu_queries_on(T, x, 4097)
        itp = linear_interp(x, y; extrap = ClampExtrap())
        ditp = adapt(MtlArray, itp)
        @test ditp.x === itp.x
        @test Metal.mtlconvert(ditp).x === itp.x
        @test !any(==(Float64), fieldtypes(typeof(itp.x)))
        for (e, q) in ((ClampExtrap(), q_oob), (FillExtrap(T(NaN)), q_oob), (InBounds(), q_in))
            itp_e = linear_interp(x, y; extrap = e)
            @test gpu_close(adapt(MtlArray, itp_e).(MtlArray(q)), itp_e.(q), ymax)
        end
    end

    for (gname, x) in gpu_grids(T, 65)
        y = gpu_fy.(x)
        ymax = maximum(abs, y)
        q_in, q_oob = gpu_queries(T, 4097)
        itp = linear_interp(x, y; extrap = ClampExtrap())

        @testset "$gname: kernel argument is isbits and holds device leaves" begin
            k = Metal.mtlconvert(adapt(MtlArray, itp))
            @test isbits(k)
            @test k.y isa FI._DeviceArray{T, 1, <:MtlDeviceArray}
            gname == "Vector" && @test k.x isa FI._DeviceArray{T, 1, <:MtlDeviceArray}
            @test isbits(Metal.mtlconvert(deriv1(adapt(MtlArray, itp))))
        end

        @testset "$gname: broadcast parity ($ename)" for (ename, e, qk) in gpu_extraps(T)
            itp_e = linear_interp(x, y; extrap = e)
            ditp = adapt(MtlArray, itp_e)
            q = qk === :in ? q_in : q_oob
            @test gpu_close(ditp.(MtlArray(q)), itp_e.(q), ymax)
            dcpu = deriv1(itp_e).(q)
            @test gpu_close(deriv1(ditp).(MtlArray(q)), dcpu, maximum(abs, filter(!isnan, dcpu)))
        end

        @testset "$gname: batch call parity ($ename)" for (ename, e, qk) in (gpu_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            itp_e = linear_interp(x, y; extrap = e)
            ditp = adapt(MtlArray, itp_e)
            q = qk === :in ? q_in : q_oob
            qd = MtlArray(q)
            @test gpu_close(ditp(qd), itp_e(q), ymax)
            out = similar(qd)
            @test ditp(out, qd) === out
            @test gpu_close(out, itp_e(q), ymax)
            dcpu = itp_e(q; deriv = DerivOp(1))
            @test gpu_close(ditp(qd; deriv = DerivOp(1)), dcpu, maximum(abs, filter(!isnan, dcpu)))
        end

        # The device-side domain check compiles on Metal; on failure the host formats the
        # CPU error from two endpoint copies, so bounds and message match the CPU.
        @testset "$gname: batch NoExtrap OOB throws the CPU error" begin
            itp_n = linear_interp(x, y)
            cpu_err = try
                itp_n(q_oob)
            catch err
                err
            end
            gpu_err = try
                adapt(MtlArray, itp_n)(MtlArray(q_oob))
            catch err
                err
            end
            @test gpu_err isa DomainError
            @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)
        end

        @testset "$gname: strided GPU view queries" begin
            qd = MtlArray(q_oob)
            v = view(qd, 1:2:length(qd))
            @test gpu_close(adapt(MtlArray, itp)(v), itp(q_oob[1:2:end]), ymax)
        end

        # Built straight from Metal arrays: same storage as a CPU build moved with `adapt`.
        @testset "$gname: built from device data" begin
            xd = gname == "Range" ? x : MtlArray(collect(x))
            bitp = linear_interp(xd, MtlArray(y); extrap = ClampExtrap())
            @test typeof(bitp) === typeof(adapt(MtlArray, itp))
            @test gpu_close(bitp(MtlArray(q_oob)), itp(q_oob), ymax)
            @test gpu_close(bitp.(MtlArray(q_oob)), itp.(q_oob), ymax)
            iitp = linear_interp(xd, MtlArray(Int32.(1:length(x))); extrap = ClampExtrap())
            @test iitp.y isa FI._DeviceVector{T, <:MtlArray}               # promoted on the device
            @test gpu_close(iitp(MtlArray(q_oob)), linear_interp(x, Int32.(1:length(x)); extrap = ClampExtrap())(q_oob), T(length(x)))
        end

        @testset "$gname: one-shot on device data ($ename)" for (ename, e, qk) in (gpu_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            q = qk === :in ? q_in : q_oob
            xd = gname == "Range" ? x : MtlArray(collect(x))
            ref = linear_interp(x, y, q; extrap = e)
            @test gpu_close(linear_interp(xd, MtlArray(y), MtlArray(q); extrap = e), ref, ymax)
            out = MtlArray(similar(q))
            @test linear_interp!(out, xd, MtlArray(y), MtlArray(q); extrap = e) === out
            @test gpu_close(out, ref, ymax)
        end

        # Periodic: WrapExtrap compiles in kernels; device data is prepared on the device.
        @testset "$gname: periodic $bname" for bname in ("inclusive", "exclusive")
            qp = T(3.6) .* q_in .- T(1.3)
            if bname == "inclusive"
                xs, ys, bc = x, sin.(2 * T(π) .* x), PeriodicBC()
            else
                xs, ys, bc = x[1:(end - 1)], sin.(2 * T(π) .* x[1:(end - 1)]), PeriodicBC(endpoint = :exclusive, period = one(T))
            end
            ref = linear_interp(xs, ys; bc)
            cpu = ref(qp)
            @test gpu_close(adapt(MtlArray, ref)(MtlArray(qp)), cpu, one(T))
            @test gpu_close(adapt(MtlArray, ref).(MtlArray(qp)), cpu, one(T))
            xd = gname == "Range" ? xs : MtlArray(collect(xs))
            @test gpu_close(linear_interp(xd, MtlArray(ys); bc)(MtlArray(qp)), cpu, one(T))
            @test gpu_close(linear_interp(xd, MtlArray(ys), MtlArray(qp); bc), linear_interp(xs, ys, qp; bc), one(T))
        end

        @testset "$gname: stored hint-carrying policy compiles ($(nameof(typeof(s))))" for s in (LinearBinarySearch(), LinearSearch())
            itp_s = linear_interp(x, y; extrap = ClampExtrap(), search = s)
            @test gpu_close(adapt(MtlArray, itp_s).(MtlArray(q_oob)), itp_s.(q_oob), ymax)
        end

        @testset "$gname: ~10^6 queries" begin
            qbig = repeat(q_oob, 245)
            @test gpu_close(adapt(MtlArray, itp).(MtlArray(qbig)), itp.(qbig), ymax)
        end

        # `mtl` narrows Float64 storage to Float32: grid, data and fill value follow.
        @testset "$gname: mtl(Float64 interpolant)" for e in (ClampExtrap(), FillExtrap(NaN))
            x64 = Float64.(x)
            itp64 = linear_interp(gname == "Range" ? range(0.0, 1.0; length = length(x)) : x64, gpu_fy.(x64); extrap = e)
            ditp = mtl(itp64)
            @test eltype(ditp.x) === Float32
            @test eltype(ditp.y) === Float32
            ref = linear_interp(gname == "Range" ? range(0.0f0, 1.0f0; length = length(x)) : x, y; extrap = e)
            @test gpu_close(ditp.(MtlArray(q_oob)), ref.(q_oob), ymax)
        end

        # Endpoint reads in the Clamp/Fill OOB branches are unchecked through the device
        # leaf (no backend registration): a bounds-checked read costs ≈ 100+ µs per launch.
        @testset "$gname: unchecked endpoint reads without a backend extension" begin
            ditp = adapt(MtlArray, itp)
            raw = FI.LinearInterpolant(FI._RawFields(), ditp.x, parent(ditp.y), ditp.extrap, ditp.search_policy)
            qd = MtlArray(q_oob)
            out = similar(qd)
            t_leaf = device_time(() -> (out .= ditp.(qd)))
            t_raw = device_time(() -> (out .= raw.(qd)))       # raw MtlArray y: checked first/last
            @test t_leaf < t_raw
        end

        # The NoExtrap OOB message is built with String interpolation, which Metal cannot
        # compile; the batch call (domain check outside the evaluation kernel) is the route.
        @testset "$gname: NoExtrap broadcast is a known GPU compile failure" begin
            ditp = adapt(MtlArray, linear_interp(x, y))
            err = try
                ditp.(MtlArray(q_in))
                nothing
            catch e
                e
            end
            @test err !== nothing && is_invalid_ir(err)
        end

        # Metal flushes subnormals to zero: ±nextfloat(0) evaluate at 0, and the negative one
        # is classified in-domain (the CPU fills it with NaN).
        @testset "$gname: subnormal queries flush to zero" begin
            itp_f = linear_interp(x, y; extrap = FillExtrap(T(NaN)))
            dev = Array(adapt(MtlArray, itp_f).(MtlArray(T[nextfloat(zero(T)), prevfloat(zero(T))])))
            @test dev[1] == itp_f(zero(T))
            @test dev[2] == itp_f(zero(T))
            @test isnan(itp_f(prevfloat(zero(T))))
        end
    end
end
