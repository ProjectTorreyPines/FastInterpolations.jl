# ========================================
# GPU storage for 1D LinearInterpolant (JLArrays)
# ========================================
# JLArrays is a CPU-backed GPUArrays emulator: it checks the device storage leaf
# (`_DeviceArray`), the Adapt rebuild contracts and the absence of host scalar indexing,
# but runs kernels as plain Julia, so GPU *compilation* is covered by the local Metal test
# test/gpu/metal_linear1d.jl.

using Test
using FastInterpolations
using Adapt, GPUArrays, JLArrays
const FI = FastInterpolations

include(joinpath(@__DIR__, "..", "gpu", "linear1d_cases.jl"))

# Same-type copying adaptor: adapt must rebuild from the adapted fields, never discard them
# because their types did not change.
struct _CopyAdaptor end
Adapt.adapt_storage(::_CopyAdaptor, x::Array) = copy(x)

# Host precision-changing adaptor, as `Metal.mtl` / `CUDA.cu` do on their way to the device.
struct _Float32Adaptor end
Adapt.adapt_storage(::_Float32Adaptor, x::Array{Float64}) = Float32.(x)

@testset "GPU storage: LinearInterpolant 1D (JLArrays)" begin
    GPUArrays.allowscalar(false)
    adapt_ext = Base.get_extension(FastInterpolations, :FastInterpolationsAdaptExt)
    gpu_ext = Base.get_extension(FastInterpolations, :FastInterpolationsGPUArraysExt)
    @test adapt_ext !== nothing
    @test gpu_ext !== nothing

    # The raw constructor and the Adapt rule enumerate these fields.
    @test fieldnames(FI.LinearInterpolant) == (:x, :y, :extrap, :search_policy)

    @testset "_DeviceArray leaf" begin
        a = JLArray(Float32[1, 2, 3])
        d = FI._DeviceArray(a)
        @test d isa FI._DeviceVector{Float32}
        @test d isa AbstractVector{Float32}
        @test parent(d) === a
        @test size(d) == (3,)
        @test FI._DeviceArray(JLArray(zeros(Float32, 2, 3))) isa AbstractMatrix{Float32}
        @test occursin("_DeviceArray", sprint(show, d))                       # no element reads
        @test occursin("_DeviceArray", sprint(show, MIME"text/plain"(), d))
    end

    for T in (Float32, Float64), (gname, x) in gpu_grids(T, 33)
        y = gpu_fy.(x)
        ymax = maximum(abs, y)
        q_in, q_oob = gpu_queries(T, 257)
        itp = linear_interp(x, y; extrap = ClampExtrap())

        @testset "$T $gname: device form" begin
            ditp = adapt(JLArray, itp)
            @test ditp.y isa FI._DeviceVector{T}
            @test parent(ditp.y) isa JLArray
            if gname == "Range"
                @test ditp.x === itp.x                                    # isbits grid by value
            else
                @test ditp.x isa FI._DeviceVector{T}                      # no spacing caches
                @test parent(ditp.x) isa JLArray
            end
            @test ditp.extrap === itp.extrap
            @test ditp.search_policy === itp.search_policy
            @test occursin("LinearInterpolant", sprint(show, MIME"text/plain"(), ditp))
        end

        @testset "$T $gname: host adaptation is an identity" begin
            @test adapt(Array, itp) === itp
        end

        @testset "$T $gname: same-type copying adaptor rebuilds" begin
            citp = adapt(_CopyAdaptor(), itp)
            @test citp !== itp
            @test citp.y !== itp.y
            @test citp.y == itp.y
            @test typeof(citp.x) === typeof(itp.x)                        # stays host storage
            @test citp(T(0.3)) == itp(T(0.3))
        end

        @testset "$T $gname: kernel conversion keeps the leaf" begin
            kitp = JLArrays.jlconvert(adapt(JLArray, itp))
            @test kitp.y isa FI._DeviceArray
            @test !(parent(kitp.y) isa JLArray)                           # device view, not the handle
            if gname == "Vector"
                @test kitp.x isa FI._DeviceArray
                @test !(parent(kitp.x) isa JLArray)
            end
        end

        @testset "$T $gname: broadcast parity ($ename)" for (ename, e, qk) in gpu_extraps(T)
            itp_e = linear_interp(x, y; extrap = e)
            ditp = adapt(JLArray, itp_e)
            q = qk === :in ? q_in : q_oob
            @test gpu_close(ditp.(JLArray(q)), itp_e.(q), ymax)
            dcpu = deriv1(itp_e).(q)
            @test gpu_close(deriv1(ditp).(JLArray(q)), dcpu, maximum(abs, filter(!isnan, dcpu)))
        end

        @testset "$T $gname: round trip restores host storage" begin
            back = adapt(Array, adapt(JLArray, itp))
            @test back.y isa Vector{T}
            @test typeof(back.x) === typeof(itp.x)                        # `_CachedVector` rebuilt
            @test back.(q_oob) == itp.(q_oob)
        end

        @testset "$T $gname: DerivativeView adapts its parent" begin
            d = adapt(JLArray, deriv1(itp))
            @test d isa FI.DerivativeView{1}
            @test d.parent.y isa FI._DeviceVector{T}
        end

        @testset "$T $gname: batch call ($ename)" for (ename, e, qk) in (gpu_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
            itp_e = linear_interp(x, y; extrap = e)
            ditp = adapt(JLArray, itp_e)
            q = qk === :in ? q_in : q_oob
            qd = JLArray(q)
            res = ditp(qd)
            @test res isa JLArray{T}
            @test gpu_close(res, itp_e(q), ymax)
            out = similar(qd)
            @test ditp(out, qd) === out
            @test gpu_close(out, itp_e(q), ymax)
            dcpu = itp_e(q; deriv = DerivOp(1))
            @test gpu_close(ditp(qd; deriv = DerivOp(1)), dcpu, maximum(abs, filter(!isnan, dcpu)))
        end

        @testset "$T $gname: batch planning" begin
            itp_n = linear_interp(x, y)                                   # NoExtrap
            ditp = adapt(JLArray, itp_n)
            cpu_err = try
                itp_n(q_oob)
            catch err
                err
            end
            gpu_err = try
                ditp(JLArray(q_oob))
            catch err
                err
            end
            @test gpu_err isa DomainError
            @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)  # same bounds and message
            @test gpu_close(ditp(JLArray(q_in); extrap = InBounds()), itp_n(q_in), ymax)
            @test_throws ArgumentError itp_n(q_oob; extrap = ClampExtrap())        # override contract:
            @test_throws ArgumentError ditp(JLArray(q_oob); extrap = ClampExtrap()) # InBounds only, as on the CPU
            @test_throws ArgumentError ditp(JLArray(q_in); hint = Ref(1))
            @test_throws DimensionMismatch ditp(JLArray(zeros(T, 3)), JLArray(q_in))
            @test length(ditp(JLArray(T[]))) == 0
        end

        @testset "$T $gname: strided GPU views" begin
            ditp = adapt(JLArray, itp)
            qd = JLArray(q_oob)
            v = view(qd, 1:2:length(qd))
            @test gpu_close(ditp(v), itp(q_oob[1:2:end]), ymax)
            out = JLArray(zeros(T, 2 * length(v)))
            ov = view(out, 2:2:length(out))
            @test ditp(ov, v) === ov
            @test gpu_close(Array(out)[2:2:end], itp(q_oob[1:2:end]), ymax)
        end
    end

    # Uniform grids need no device storage: the isbits `_CachedRange` travels by value into
    # kernels for every range input (incl. Int `1:n`, unit-step search path).
    @testset "$T range grid ($rname) stays a range on the device" for T in (Float32, Float64), (rname, x) in gpu_range_grids(T, 65)
        y = gpu_values(T, x)
        ymax = maximum(abs, y)
        q_in, q_oob = gpu_queries_on(T, x, 257)
        itp = linear_interp(x, y; extrap = ClampExtrap())
        ditp = adapt(JLArray, itp)
        @test ditp.x === itp.x
        @test JLArrays.jlconvert(ditp).x === itp.x
        for (e, q) in ((ClampExtrap(), q_oob), (FillExtrap(T(NaN)), q_oob), (InBounds(), q_in))
            itp_e = linear_interp(x, y; extrap = e)
            @test gpu_close(adapt(JLArray, itp_e).(JLArray(q)), itp_e.(q), ymax)
        end
    end

    # A precision-changing move narrows the grid together with the data and the fill value.
    @testset "precision-changing adaptor ($gname)" for (gname, x) in gpu_grids(Float64, 33)
        y = gpu_fy.(x)
        for e in (ClampExtrap(), FillExtrap(NaN))
            itp32 = adapt(_Float32Adaptor(), linear_interp(x, y; extrap = e))
            @test eltype(itp32.y) === Float32
            @test eltype(itp32.x) === Float32
            @test gname != "Range" || itp32.x isa FI._CachedRange
            @test gname != "Vector" || itp32.x isa FI._CachedVector       # caches rebuilt
            @test !(itp32.extrap isa FillExtrap) || itp32.extrap.fill_value isa Float32
            ref = linear_interp(Float32.(x), Float32.(y); extrap = e)
            q = gpu_queries(Float32, 257)[2]
            @test gpu_close(itp32.(q), ref.(q), 2.0f0)
        end
    end

    # Built straight from device data: the same storage as a CPU build moved with `adapt`,
    # no host round trip (allowscalar(false) would reject any element read).
    @testset "$T $gname: built from device data" for T in (Float32, Float64), (gname, x) in gpu_grids(T, 33)
        y = gpu_fy.(x)
        ymax = maximum(abs, y)
        q_oob = gpu_queries(T, 257)[2]
        xd = gname == "Range" ? x : JLArray(collect(x))
        yd = JLArray(y)
        ref = linear_interp(x, y; extrap = ClampExtrap())
        bitp = linear_interp(xd, yd; extrap = ClampExtrap())
        @test typeof(bitp) === typeof(adapt(JLArray, ref))
        @test parent(bitp.y) !== yd                                       # default store copies
        @test gpu_close(bitp(JLArray(q_oob)), ref(q_oob), ymax)
        @test gpu_close(bitp.(JLArray(q_oob)), ref.(q_oob), ymax)
        @test typeof(adapt(Array, bitp)) === typeof(ref)                  # back to host storage
        # reference store aliases the caller's device arrays
        ritp = linear_interp(xd, yd; extrap = ClampExtrap(), store = StorePolicy(copy = false))
        @test parent(ritp.y) === yd
        gname == "Vector" && @test parent(ritp.x) === xd
        # an uncached axis is still device storage
        uitp = linear_interp(xd, yd; extrap = ClampExtrap(), store = StorePolicy(cache_axis = false))
        @test uitp.x isa (gname == "Range" ? FI._CachedRange : FI._DeviceVector{T})
        @test gpu_close(uitp(JLArray(q_oob)), ref(q_oob), ymax)
    end

    @testset "integer device data is promoted on the device" begin
        x = JLArray(collect(range(0.0f0, 1.0f0; length = 9)))
        y = JLArray(Int32.(0:8))
        itp = linear_interp(x, y; extrap = ClampExtrap())
        @test itp.y isa FI._DeviceVector{Float32}
        @test parent(itp.y) isa JLArray
        @test Array(itp(JLArray(Float32[0.0625, 0.5]))) ≈ Float32[0.5, 4.0]
    end

    @testset "batch admission and storage checks" begin
        x = range(0.0f0, 1.0f0; length = 33)
        y = gpu_fy.(x)
        q = gpu_queries(Float32, 64)[1]
        qd = JLArray(q)
        itp = linear_interp(x, y)
        ditp = adapt(JLArray, itp)
        @test_throws ArgumentError itp(qd)                                 # data still on the host
        @test_throws ArgumentError ditp(Vector{Float32}(undef, 64), qd)    # mixed output / query storage
        @test_throws ArgumentError ditp(similar(qd), q)
        citp = cubic_interp(x, y)
        @test_throws ArgumentError citp(qd)                                # family not admitted
        @test_throws ArgumentError citp(JLArray(Float32[]))                # ... checked before the empty shortcut
        dref = deriv1(itp)(q)
        @test gpu_close(deriv1(ditp)(qd), dref, maximum(abs, dref))        # DerivativeView → GPU batch path
        # Built from device data (range grid): already device-resident, evaluated in place.
        bitp = linear_interp(x, JLArray(y))
        @test gpu_close(bitp(qd), itp(q), maximum(abs, y))
    end

    @testset "no method ambiguities from the extensions" begin
        @test isempty(Test.detect_ambiguities(adapt_ext, gpu_ext))
    end
end
