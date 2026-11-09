# ========================================
# Real-GPU tests: 1D LinearInterpolant (included by test/gpu/runtests.jl)
# ========================================
# What the JLArrays extension test cannot cover: real kernel compilation and device numerics.
# Backend-agnostic: arrays move with `dev` (KernelAbstractions backend) or
# `adapt(GPUArrayType, ·)`; every supported float type runs.

using Test
using FastInterpolations
using Adapt, GPUArrays, ForwardDiff, StaticArrays, Unitful
using InteractiveUtils: code_llvm
using KernelAbstractions: @kernel, @index

include(joinpath(@__DIR__, "linear1d_cases.jl"))

is_invalid_ir(err) = nameof(typeof(err)) === :InvalidIRError

@kernel function _eval_each!(out, f, q)
    i = @index(Global, Linear)
    @inbounds out[i] = f(q[i])
end

@testset "$GPU_NAME: LinearInterpolant 1D ($T)" for T in GPU_ELTYPES
    GPUArrays.allowscalar(false)

    @testset "extensions load; no backend-specific extension" begin
        @test Base.get_extension(FastInterpolations, :FastInterpolationsAdaptExt) !== nothing
        @test Base.get_extension(FastInterpolations, :FastInterpolationsGPUArraysExt) !== nothing
        @test Base.get_extension(FastInterpolations, Symbol(:FastInterpolations, GPU_SPEC.pkg, :Ext)) === nothing
    end

    @testset "batch admission: host data and unsupported families are rejected" begin
        x = range(zero(T), one(T); length = 33)
        q = dev(gpu_queries(T, 64)[1])
        @test_throws ArgumentError linear_interp(x, gpu_fy.(x))(q)       # not on the device yet
        @test_throws ArgumentError cubic_interp(x, gpu_fy.(x))(q)        # family not admitted
    end

    @testset "persistent shaped queries keep their shape ($gname)" for (gname, x) in gpu_grids(T, 33)
        itp = linear_interp(x, gpu_fy.(x); extrap = ClampExtrap())
        ditp = adapt(GPUArrayType, itp)
        qm = reshape(gpu_queries(T, 4096)[2], 64, 64)
        rm = ditp(dev(qm))
        @test rm isa AbstractArray{T, 2} && on_device(rm)
        @test gpu_close(rm, itp(qm), T(2))
        dm = itp(qm; deriv = DerivOp(1))
        @test gpu_close(ditp(dev(qm); deriv = DerivOp(1)), dm, maximum(abs, dm))
    end

    # Any isbits element compiles: the same evaluation code, specialized per element type.
    @testset "element types: SVector / Complex values, Dual / Unitful queries" begin
        x = range(zero(T), one(T); length = 65)
        q = gpu_queries(T, 4097)[2]
        for y in ([SVector(sin(3t), cos(2t)) for t in x], [cis(3t) for t in x])
            itp = linear_interp(x, y; extrap = ClampExtrap())
            ditp = adapt(GPUArrayType, itp)
            @test Array(ditp(dev(q))) ≈ itp.(q)
            @test Array(ditp.(dev(q))) ≈ itp.(q)
            @test Array(ditp(dev(q); deriv = DerivOp(1))) ≈ deriv1(itp).(q)
        end
        itp = linear_interp(x, gpu_fy.(x); extrap = ClampExtrap())
        r = Array(adapt(GPUArrayType, itp)(dev(ForwardDiff.Dual.(q, one(T)))))
        @test ForwardDiff.value.(r) ≈ itp.(q)
        @test ForwardDiff.partials.(r, 1) ≈ deriv1(itp).(q)
        xu, yu, qu = collect(x) .* u"m", gpu_fy.(collect(x)) .* u"T", gpu_queries(T, 4097)[1] .* u"m"
        itpu = linear_interp(xu, yu; extrap = ClampExtrap())
        ref = itpu.(qu)
        @test Array(adapt(GPUArrayType, itpu)(dev(qu))) ≈ ref
        @test Array(linear_interp(dev(xu), dev(yu); extrap = ClampExtrap())(dev(qu))) ≈ ref
        @test Array(linear_interp(dev(xu), dev(yu), dev(qu); extrap = ClampExtrap())) ≈ ref
        @test Array(adapt(GPUArrayType, linear_interp(xu, yu))(dev(qu))) ≈ ref
    end

    # Uniform grids: the isbits `_CachedRange` stays a range inside kernels, with fields of the
    # grid's float type only, even for Base `range` input (Float64 internals) and Int `1:n`.
    @testset "range grid ($rname) stays a range on the device" for (rname, x) in gpu_range_grids(T, 65)
        y = gpu_values(T, x)
        ymax = maximum(abs, y)
        q_in, q_oob = gpu_queries_on(T, x, 4097)
        itp = linear_interp(x, y; extrap = ClampExtrap())
        ditp = adapt(GPUArrayType, itp)
        @test ditp.x === itp.x
        @test kernel_arg(ditp).x === itp.x
        @test all(t -> !(t <: AbstractFloat) || t === T, fieldtypes(typeof(itp.x)))
        for (e, q) in ((ClampExtrap(), q_oob), (FillExtrap(T(NaN)), q_oob), (InBounds(), q_in))
            itp_e = linear_interp(x, y; extrap = e)
            @test gpu_close(adapt(GPUArrayType, itp_e).(dev(q)), itp_e.(q), ymax)
        end
    end

    for (gname, x) in gpu_grids(T, 65)
        y = gpu_fy.(x)
        ymax = maximum(abs, y)
        q_in, q_oob = gpu_queries(T, 4097)
        itp = linear_interp(x, y; extrap = ClampExtrap())

        @testset "$gname: kernel argument is isbits and holds device leaves" begin
            k = kernel_arg(adapt(GPUArrayType, itp))
            @test isbits(k)
            @test k.y isa FI._DeviceArray{T, 1} && isbits(parent(k.y))
            gname == "Vector" && @test k.x isa FI._DeviceArray{T, 1} && isbits(parent(k.x))
            @test isbits(kernel_arg(deriv1(adapt(GPUArrayType, itp))))
        end

        @testset "$gname: broadcast parity ($ename)" for (ename, e, qk) in gpu_extraps(T)
            itp_e = linear_interp(x, y; extrap = e)
            ditp = adapt(GPUArrayType, itp_e)
            q = qk === :in ? q_in : q_oob
            @test gpu_close(ditp.(dev(q)), itp_e.(q), ymax)
            dcpu = deriv1(itp_e).(q)
            @test gpu_close(deriv1(ditp).(dev(q)), dcpu, maximum(abs, filter(!isnan, dcpu)))
        end

        @testset "$gname: batch call parity ($ename)" for (ename, e, qk) in (gpu_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            itp_e = linear_interp(x, y; extrap = e)
            ditp = adapt(GPUArrayType, itp_e)
            q = qk === :in ? q_in : q_oob
            qd = dev(q)
            @test gpu_close(ditp(qd), itp_e(q), ymax)
            out = similar(qd)
            @test ditp(out, qd) === out
            @test gpu_close(out, itp_e(q), ymax)
            dcpu = itp_e(q; deriv = DerivOp(1))
            @test gpu_close(ditp(qd; deriv = DerivOp(1)), dcpu, maximum(abs, filter(!isnan, dcpu)))
        end

        # The device-side domain check compiles; on failure the host formats the CPU error
        # from two endpoint copies, so bounds and message match the CPU.
        @testset "$gname: batch NoExtrap OOB throws the CPU error" begin
            itp_n = linear_interp(x, y)
            cpu_err = try
                itp_n(q_oob)
            catch err
                err
            end
            gpu_err = try
                adapt(GPUArrayType, itp_n)(dev(q_oob))
            catch err
                err
            end
            @test gpu_err isa DomainError
            @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)
            # NaN queries are rejected by the in-kernel check as on the CPU
            qn = T[q_in[1], T(NaN)]
            nan_err(f) = try
                f()
                ""
            catch err
                sprint(showerror, err)
            end
            @test nan_err(() -> adapt(GPUArrayType, itp_n)(dev(qn))) == nan_err(() -> itp_n(qn))
        end

        @testset "$gname: strided GPU view queries" begin
            qd = dev(q_oob)
            v = view(qd, 1:2:length(qd))
            @test gpu_close(adapt(GPUArrayType, itp)(v), itp(q_oob[1:2:end]), ymax)
        end

        # Built straight from device arrays: same storage as a CPU build moved with `adapt`.
        @testset "$gname: built from device data" begin
            xd = gname == "Range" ? x : dev(collect(x))
            bitp = linear_interp(xd, dev(y); extrap = ClampExtrap())
            @test typeof(bitp) === typeof(adapt(GPUArrayType, itp))
            @test gpu_close(bitp(dev(q_oob)), itp(q_oob), ymax)
            @test gpu_close(bitp.(dev(q_oob)), itp.(q_oob), ymax)
            # the default store owns copies: in-place changes to the caller's arrays stay out
            xo, yo = gname == "Range" ? x : dev(collect(x)), dev(y)
            oitp = linear_interp(xo, yo; extrap = ClampExtrap())
            before = Array(oitp(dev(q_oob)))
            yo .*= 10
            gname == "Range" || (xo .*= T(0.5))
            @test Array(oitp(dev(q_oob))) == before
            iitp = linear_interp(xd, dev(Int32.(1:length(x))); extrap = ClampExtrap())
            @test iitp.y isa FI._DeviceVector{T} && on_device(parent(iitp.y))   # promoted on the device
            @test gpu_close(iitp(dev(q_oob)), linear_interp(x, Int32.(1:length(x)); extrap = ClampExtrap())(q_oob), T(length(x)))
        end

        @testset "$gname: one-shot on device data ($ename)" for (ename, e, qk) in (gpu_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            q = qk === :in ? q_in : q_oob
            xd = gname == "Range" ? x : dev(collect(x))
            ref = linear_interp(x, y, q; extrap = e)
            @test gpu_close(linear_interp(xd, dev(y), dev(q); extrap = e), ref, ymax)
            out = dev(similar(q))
            @test linear_interp!(out, xd, dev(y), dev(q); extrap = e) === out
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
            @test gpu_close(adapt(GPUArrayType, ref)(dev(qp)), cpu, one(T))
            @test gpu_close(adapt(GPUArrayType, ref).(dev(qp)), cpu, one(T))
            xd = gname == "Range" ? xs : dev(collect(xs))
            @test gpu_close(linear_interp(xd, dev(ys); bc)(dev(qp)), cpu, one(T))
            @test gpu_close(linear_interp(xd, dev(ys), dev(qp); bc), linear_interp(xs, ys, qp; bc), one(T))
        end

        @testset "$gname: stored hint-carrying policy compiles ($(nameof(typeof(s))))" for s in (LinearBinarySearch(), LinearSearch())
            itp_s = linear_interp(x, y; extrap = ClampExtrap(), search = s)
            @test gpu_close(adapt(GPUArrayType, itp_s).(dev(q_oob)), itp_s.(q_oob), ymax)
        end

        @testset "$gname: ~10^6 queries" begin
            qbig = repeat(q_oob, 245)
            @test gpu_close(adapt(GPUArrayType, itp).(dev(qbig)), itp.(qbig), ymax)
        end

        # The backend's precision-changing converter (`cu`, `mtl`) on a Float64 interpolant:
        # the grid converts as the converter converts a vector grid, the values and the fill
        # value as it converts them (Complex and static vector elements included).
        if T === Float32 && NARROW !== nothing
            @testset "$gname: $(GPU_SPEC.narrow)(Float64 interpolant), $vname values" for (vname, fv, fill) in (
                    ("Float64", t -> sin(3t), NaN),
                    ("Complex", t -> cis(3t), complex(NaN)),
                    ("SVector", t -> SVector(sin(3t), cos(2t)), SVector(NaN, NaN)),
                )
                x64 = gname == "Range" ? range(0.0, 1.0; length = length(x)) : Float64.(x)
                y64 = fv.(x64)
                itp64 = linear_interp(x64, y64; extrap = FillExtrap(fill))
                ditp = NARROW(itp64)
                @test eltype(ditp.x) === eltype(NARROW(collect(x64)))
                @test eltype(ditp.y) === eltype(NARROW(y64))
                @test typeof(ditp.extrap.fill_value) === eltype(ditp.y)
                @test isapprox(Array(ditp.(dev(q_oob))), itp64.(q_oob); rtol = 1.0e-5, nans = true)
                @test isapprox(Array(ditp(dev(q_oob))), itp64.(q_oob); rtol = 1.0e-5, nans = true)
            end
        end

        # Endpoint reads in the Clamp/Fill OOB branches are unchecked through the device leaf
        # (a bounds-checked read measured ≈ 100+ µs per launch on Metal): no bounds-error
        # path in the code a kernel gets, unlike the raw device array it wraps. A process
        # started with --check-bounds=yes (Pkg.test's default) keeps every check.
        @testset "$gname: unchecked endpoint reads" begin
            leaf = kernel_arg(adapt(GPUArrayType, itp)).y
            checks_bounds(f, a) = occursin("throw_boundserror", sprint(io -> code_llvm(io, f, Tuple{typeof(a)}; debuginfo = :none)))
            forced = Base.JLOptions().check_bounds          # 0 default, 1 yes, 2 no
            @test checks_bounds(first, leaf) == (forced == 1)
            @test checks_bounds(last, leaf) == (forced == 1)
            @test checks_bounds(first, parent(leaf)) == (forced != 2)
        end

        @testset "$gname: batch kernel inlines the point evaluation ($ename)" for (ename, e) in (("Clamp", ClampExtrap()), ("NoExtrap", NoExtrap()))
            ditp = adapt(GPUArrayType, linear_interp(x, y; extrap = e))
            q = dev(q_in)
            @test !calls_ext_function(device_typed(() -> ditp(similar(q), q)))
            @test !calls_ext_function(device_typed(() -> linear_interp!(similar(q), dev(collect(x)), dev(y), q; extrap = e)))
        end

        # The NoExtrap OOB message is built with String interpolation, which GPU compilers
        # reject: a NoExtrap broadcast over device queries is the batch call (domain check
        # outside the evaluation kernel); inside a user kernel it fails to compile.
        @testset "$gname: NoExtrap broadcast takes the batch call" begin
            itp_n = linear_interp(x, y)
            ditp = adapt(GPUArrayType, itp_n)
            @test gpu_close(ditp.(dev(q_in)), itp_n.(q_in), ymax)
            @test gpu_close(2 .* ditp.(dev(q_in)), 2 .* itp_n.(q_in), 2 * ymax)
            @test gpu_close(ditp.(T(0.5) .* dev(q_in)), itp_n.(T(0.5) .* q_in), ymax)   # nested: materialized first
            errmsg(f) = try
                f()
                ""
            catch err
                sprint(showerror, err)
            end
            @test errmsg(() -> ditp.(dev(q_oob))) == errmsg(() -> itp_n(q_oob))   # the batch call's error
        end

        @testset "$gname: NoExtrap inside a user kernel is a known GPU compile failure" begin
            ditp = adapt(GPUArrayType, linear_interp(x, y))
            qd = dev(q_in)
            err = try
                _eval_each!(BACKEND)(similar(qd), ditp, qd; ndrange = length(qd))
                nothing
            catch e
                e
            end
            @test err !== nothing && is_invalid_ir(err)
        end

        # Subnormal queries: a device that flushes them to zero (Metal) evaluates ±nextfloat(0)
        # at 0 and classifies the negative one in-domain (the CPU fills it with NaN); others
        # match the CPU.
        @testset "$gname: subnormal queries" begin
            itp_f = linear_interp(x, y; extrap = FillExtrap(T(NaN)))
            qs = T[nextfloat(zero(T)), prevfloat(zero(T))]
            res = Array(adapt(GPUArrayType, itp_f).(dev(qs)))
            if FLUSHES_SUBNORMALS
                @test res[1] == itp_f(zero(T))
                @test res[2] == itp_f(zero(T))
                @test isnan(itp_f(prevfloat(zero(T))))
            else
                @test isequal(res, itp_f.(qs))
            end
        end
    end
end
