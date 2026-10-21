# ========================================
# GPU storage for 1D LinearInterpolant (JLArrays)
# ========================================
# JLArrays is a CPU-backed GPUArrays emulator: it checks the device storage leaf
# (`_DeviceArray`), the Adapt rebuild contracts and the absence of host scalar indexing,
# but runs kernels as plain Julia, so GPU *compilation* is covered by the local Metal test
# test/gpu/metal_linear1d.jl.

using Test
using FastInterpolations
using Adapt, GPUArrays, JLArrays, ForwardDiff, StaticArrays, Unitful
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
            cpu_err = try
                itp_n(zeros(T, 3), q_in)
            catch err
                err
            end
            gpu_err = try
                ditp(JLArray(zeros(T, 3)), JLArray(q_in))
            catch err
                err
            end
            @test typeof(gpu_err) === typeof(cpu_err)                         # output length: the CPU's error
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

    # Every range tag narrows and keeps its tag. `_WidenedDomain` (what x86_64 makes of a
    # Float64 `range`) is built directly so that every architecture runs it.
    @testset "precision-changing adaptor keeps the range tag ($(nameof(Tag)))" for (Tag, x) in (
            (FI._UnitStep, 1:33),
            (FI._OneTo, Base.OneTo(33)),
            (FI._WidenedDomain, FI._cached_range(FI._WidenedDomain(), 0.0, 1.0, 1 / 32, 32.0, 33)),
        )
        y = gpu_values(Float64, x)
        itp = linear_interp(x, y; extrap = ClampExtrap())
        @test itp.x isa FI._CachedRange{Float64, Float64, Tag}
        itp32 = adapt(_Float32Adaptor(), itp)
        @test itp32.x isa FI._CachedRange{Float32, Float32, Tag}
        @test eltype(itp32.y) === Float32
        ref = linear_interp(Float32.(collect(x)), Float32.(y); extrap = ClampExtrap())
        q = gpu_queries_on(Float32, x, 257)[2]
        @test gpu_close(itp32.(q), ref.(q), 2.0f0)
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

    # One-shot calls on device data: the same device paths as a persistent build.
    @testset "$T $gname: one-shot ($ename)" for T in (Float32, Float64), (gname, x) in gpu_grids(T, 33),
            (ename, e, qk) in (gpu_extraps(T)..., ("NoExtrap", NoExtrap(), :in))
        y = gpu_fy.(x)
        ymax = maximum(abs, y)
        q_in, q_oob = gpu_queries(T, 257)
        q = qk === :in ? q_in : q_oob
        xd = gname == "Range" ? x : JLArray(collect(x))
        yd, qd = JLArray(y), JLArray(q)
        res = linear_interp(xd, yd, qd; extrap = e)
        @test res isa JLArray{T}
        @test gpu_close(res, linear_interp(x, y, q; extrap = e), ymax)
        out = similar(qd)
        @test linear_interp!(out, xd, yd, qd; extrap = e) === out
        @test gpu_close(out, linear_interp(x, y, q; extrap = e), ymax)
        dcpu = linear_interp(x, y, q; extrap = e, deriv = DerivOp(1))
        @test gpu_close(linear_interp(xd, yd, qd; extrap = e, deriv = DerivOp(1)), dcpu, maximum(abs, filter(!isnan, dcpu)))
    end

    @testset "one-shot planning and admission" begin
        x = collect(range(0.0f0, 1.0f0; length = 33))
        y = gpu_fy.(x)
        q_in, q_oob = gpu_queries(Float32, 64)
        xd, yd = JLArray(x), JLArray(y)
        cpu_err = try
            linear_interp(x, y, q_oob)
        catch err
            err
        end
        gpu_err = try
            linear_interp(xd, yd, JLArray(q_oob))
        catch err
            err
        end
        @test gpu_err isa DomainError
        @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)
        @test_throws ArgumentError linear_interp(x, y, JLArray(q_in))                 # data on the host
        @test_throws ArgumentError linear_interp!(zeros(Float32, 64), xd, yd, JLArray(q_in))  # host output
        @test_throws ArgumentError linear_interp(xd, yd, JLArray(q_in); hint = Ref(1))        # no host hint
        cpu_err = try
            linear_interp(x, y, q_in; bc = PeriodicBC())
        catch err
            err
        end         # y[end] ≠ y[1]
        gpu_err = try
            linear_interp(xd, yd, JLArray(q_in); bc = PeriodicBC())
        catch err
            err
        end
        @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)
        v = view(JLArray(q_oob), 1:2:64)
        @test gpu_close(linear_interp(xd, yd, v; extrap = ClampExtrap()), linear_interp(x, y, q_oob[1:2:end]; extrap = ClampExtrap()), 2.0f0)
        # a hint-carrying search policy runs with a fresh per-thread searcher
        @test gpu_close(linear_interp(xd, yd, JLArray(q_oob); extrap = ClampExtrap(), search = LinearBinarySearch()), linear_interp(x, y, q_oob; extrap = ClampExtrap()), 2.0f0)
        # shaped GPU queries keep their shape
        qm = reshape(q_in, 8, 8)
        rm = linear_interp(xd, yd, JLArray(qm))
        @test rm isa JLArray{Float32, 2}
        @test gpu_close(rm, linear_interp(x, y, qm), 2.0f0)
    end

    # One-shot GPU calls take the core one-shot route: only the storage leaves and the
    # execution/planning seams below it are GPU methods; the API entry is the core one.
    @testset "one-shot uses the core API route" begin
        xd, yd, qd = JLArray(collect(range(0.0f0, 1.0f0; length = 9))), JLArray(rand(Float32, 9)), JLArray(rand(Float32, 4))
        @test which(linear_interp, typeof.((xd, yd, qd))).module === FastInterpolations
        @test which(linear_interp!, typeof.((similar(qd), xd, yd, qd))).module === FastInterpolations
    end

    # Persistent batch calls take the core batch route as well: the family loop is the seam.
    @testset "persistent batch uses the core API route" begin
        x = collect(range(0.0f0, 1.0f0; length = 33))
        itp = linear_interp(x, gpu_fy.(x); extrap = ClampExtrap())
        ditp = adapt(JLArray, itp)
        q = gpu_queries(Float32, 64)[2]
        qd = JLArray(q)
        @test which(ditp, Tuple{typeof(qd)}).module === FastInterpolations
        @test which(ditp, Tuple{typeof(similar(qd)), typeof(qd)}).module === FastInterpolations
        # shaped GPU queries keep their shape, allocating and in place
        qm = reshape(q, 8, 8)
        rm = ditp(JLArray(qm))
        @test rm isa JLArray{Float32, 2}
        @test gpu_close(rm, itp(qm), 2.0f0)
        om = JLArray(zeros(Float32, 8, 8))
        @test ditp(om, JLArray(qm)) === om
        @test gpu_close(om, itp(qm), 2.0f0)
        cpu_err = try
            itp(zeros(Float32, 4, 16), qm)
        catch err
            err
        end
        gpu_err = try
            ditp(JLArray(zeros(Float32, 4, 16)), JLArray(qm))
        catch err
            err
        end
        @test typeof(gpu_err) === typeof(cpu_err)                             # exact shape, as on the CPU
        itp_n = linear_interp(x, gpu_fy.(x))                                  # NoExtrap: the CPU's error
        cpu_err = try
            itp_n(qm)
        catch err
            err
        end
        gpu_err = try
            adapt(JLArray, itp_n)(JLArray(qm))
        catch err
            err
        end
        @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)
    end

    # Periodic boundary conditions: queries span several periods (WrapExtrap in the kernel).
    @testset "$T periodic $(bname) ($gname)" for T in (Float32, Float64), (gname, x) in gpu_grids(T, 33),
            bname in ("inclusive", "exclusive")
        q = T(3.6) .* gpu_queries(T, 257)[1] .- T(1.3)
        if bname == "inclusive"
            xs, ys, bc = x, sin.(2 * T(π) .* x), PeriodicBC()
        else
            xs, ys, bc = x[1:(end - 1)], sin.(2 * T(π) .* x[1:(end - 1)]), PeriodicBC(endpoint = :exclusive, period = one(T))
        end
        ref = linear_interp(xs, ys; bc)
        cpu = ref(q)
        @test gpu_close(adapt(JLArray, ref)(JLArray(q)), cpu, one(T))              # moved CPU build
        @test gpu_close(adapt(JLArray, ref).(JLArray(q)), cpu, one(T))
        xd = gname == "Range" ? xs : JLArray(collect(xs))
        bitp = linear_interp(xd, JLArray(ys); bc)                                 # built from device data
        @test typeof(bitp) === typeof(adapt(JLArray, ref))
        @test gpu_close(bitp(JLArray(q)), cpu, one(T))
        @test gpu_close(bitp(JLArray(q); deriv = DerivOp(1)), ref(q; deriv = DerivOp(1)), 8 * one(T))
        @test gpu_close(linear_interp(xd, JLArray(ys), JLArray(q); bc), linear_interp(xs, ys, q; bc), one(T))  # one-shot
    end

    @testset "periodic construction checks on device data" begin
        x = collect(range(0.0f0, 1.0f0; length = 17))
        bad = sin.(2.0f0 * Float32(π) .* x) .+ x                                  # y[end] ≠ y[1]
        cpu_err = try
            linear_interp(x, bad; bc = PeriodicBC())
        catch err
            err
        end
        gpu_err = try
            linear_interp(JLArray(x), JLArray(bad); bc = PeriodicBC())
        catch err
            err
        end
        @test gpu_err isa ArgumentError
        @test sprint(showerror, gpu_err) == sprint(showerror, cpu_err)
        @test linear_interp(JLArray(x), JLArray(bad); bc = PeriodicBC(check = false)).y isa FI._DeviceVector
        # a non-uniform exclusive grid needs an explicit period, as on the CPU
        @test_throws ArgumentError linear_interp(JLArray(x[1:(end - 1)]), JLArray(bad[1:(end - 1)]); bc = PeriodicBC(endpoint = :exclusive))
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

    # Device arrays hold any isbits element: values and queries keep the CPU element types
    # (the same evaluation code compiles for each). References are the CPU scalar path.
    @testset "element types: SVector / Complex values, Dual / Unitful queries" begin
        x = range(0.0f0, 1.0f0; length = 33)
        q = gpu_queries(Float32, 64)[2]
        for y in ([SVector(sin(3t), cos(2t)) for t in x], [cis(3t) for t in x])
            itp = linear_interp(x, y; extrap = ClampExtrap())
            ditp = adapt(JLArray, itp)
            @test Array(ditp(JLArray(q))) ≈ itp.(q)
            @test Array(ditp(JLArray(q); deriv = DerivOp(1))) ≈ deriv1(itp).(q)
        end
        itp = linear_interp(x, gpu_fy.(x); extrap = ClampExtrap())
        r = Array(adapt(JLArray, itp)(JLArray(ForwardDiff.Dual.(q, 1.0f0))))
        @test ForwardDiff.value.(r) ≈ itp.(q)
        @test ForwardDiff.partials.(r, 1) ≈ deriv1(itp).(q)
        xu, yu, qu = collect(x) .* u"m", gpu_fy.(collect(x)) .* u"T", gpu_queries(Float32, 64)[1] .* u"m"
        itpu = linear_interp(xu, yu; extrap = ClampExtrap())
        ref = itpu.(qu)
        @test Array(adapt(JLArray, itpu)(JLArray(qu))) ≈ ref
        @test Array(linear_interp(JLArray(xu), JLArray(yu); extrap = ClampExtrap())(JLArray(qu))) ≈ ref
        @test Array(linear_interp(JLArray(xu), JLArray(yu), JLArray(qu); extrap = ClampExtrap())) ≈ ref
        @test Array(adapt(JLArray, linear_interp(xu, yu))(JLArray(qu))) ≈ ref    # NoExtrap: device check in units
    end

    @testset "no method ambiguities from the extensions" begin
        @test isempty(Test.detect_ambiguities(adapt_ext, gpu_ext))
    end
end
