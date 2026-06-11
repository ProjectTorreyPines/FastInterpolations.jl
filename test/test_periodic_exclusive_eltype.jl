# Regression guard for Float32 promotion contract on `PeriodicBC{:exclusive}`
# persistent path. Historical bug: `_extend_exclusive` used x-only
# `float(eltype(x))`, silently widening `Int range × Float32 y → Float64`.

@testitem "PeriodicBC(:exclusive) joint Tg promotion — persistent path" setup = [AllocConstants] begin
    x_int = 1:6
    y_f32 = Float32[1.0, 2.0, 3.0, 4.0, 3.0, 1.0]

    @testset "Linear" begin
        itp = linear_interp(x_int, y_f32; bc = PeriodicBC(endpoint = :exclusive))
        @test eltype(itp.x) === Float32
        @test eltype(itp.y) === Float32
        @test typeof(itp(2.5f0)) === Float32
    end

    @testset "PCHIP" begin
        itp = pchip_interp(x_int, y_f32; bc = PeriodicBC(endpoint = :exclusive))
        @test eltype(itp.x) === Float32
        @test eltype(itp.y) === Float32
        @test typeof(itp(2.5f0)) === Float32
    end

    @testset "Cardinal" begin
        itp = cardinal_interp(x_int, y_f32; bc = PeriodicBC(endpoint = :exclusive))
        @test eltype(itp.x) === Float32
        @test eltype(itp.y) === Float32
        @test typeof(itp(2.5f0)) === Float32
    end

    @testset "Akima" begin
        itp = akima_interp(x_int, y_f32; bc = PeriodicBC(endpoint = :exclusive))
        @test eltype(itp.x) === Float32
        @test eltype(itp.y) === Float32
        @test typeof(itp(2.5f0)) === Float32
    end
end

@testitem "Joint Tg promotion — Float32 contract preserved across all 1D BCs" setup = [AllocConstants] begin
    # Guard against asymmetric fix — `:exclusive` must not regress `NoBC` / `:inclusive`.
    x_int = 1:6
    y_f32 = Float32[1.0, 2.0, 3.0, 4.0, 3.0, 1.0]

    methods = [
        ("Linear", (x, y; kw...) -> linear_interp(x, y; kw...)),
        ("PCHIP", (x, y; kw...) -> pchip_interp(x, y; kw...)),
        ("Cardinal", (x, y; kw...) -> cardinal_interp(x, y; kw...)),
        ("Akima", (x, y; kw...) -> akima_interp(x, y; kw...)),
    ]
    bcs = [
        ("NoBC", NoBC()),
        ("inclusive", PeriodicBC(endpoint = :inclusive)),
        ("exclusive", PeriodicBC(endpoint = :exclusive)),
    ]

    for (mname, ctor) in methods, (bname, bc) in bcs
        @testset "$mname + $bname" begin
            itp = ctor(x_int, y_f32; bc = bc)
            @test eltype(itp.x) === Float32
            @test eltype(itp.y) === Float32
        end
    end
end

@testitem "Joint Tg promotion — Float64 widening still correct" setup = [AllocConstants] begin
    # Opposite direction — `Int × Float64` must stay Float64 (not narrowed by the fix).
    x_int = 1:6
    y_f64 = Float64[1.0, 2.0, 3.0, 4.0, 3.0, 1.0]

    for ctor in (linear_interp, pchip_interp, cardinal_interp, akima_interp)
        @testset "$(nameof(ctor))" begin
            for bc in (NoBC(), PeriodicBC(endpoint = :inclusive), PeriodicBC(endpoint = :exclusive))
                itp = ctor(x_int, y_f64; bc = bc)
                @test eltype(itp.x) === Float64
                @test eltype(itp.y) === Float64
            end
        end
    end
end

@testitem "Joint Tg promotion — Float32 range × Float64 y widens to Float64" setup = [AllocConstants] begin
    # Third direction — value precision (Float64) wins over grid precision (Float32).
    x_f32 = range(0.0f0, 5.0f0; length = 6)
    y_f64 = Float64[1.0, 2.0, 3.0, 4.0, 3.0, 1.0]

    for ctor in (linear_interp, pchip_interp, cardinal_interp, akima_interp)
        @testset "$(nameof(ctor))" begin
            for bc in (NoBC(), PeriodicBC(endpoint = :inclusive), PeriodicBC(endpoint = :exclusive))
                itp = ctor(x_f32, y_f64; bc = bc)
                @test eltype(itp.x) === Float64
                @test eltype(itp.y) === Float64
            end
        end
    end
end
