@testitem "HeteroInterpolantND — spacings field removed (lock-down)" begin
    using FastInterpolations: interp, LinearInterp, PchipInterp, OnTheFly

    x = 0.0:1.0:3.0
    y = 0.0:1.0:3.0
    data = [Float64(i + j) for i in 1:4, j in 1:4]

    # OnTheFly path: HeteroInterpolantND is constructed iff coeffs == OnTheFly().
    # Use a heterogeneous method tuple (Pchip + Linear) to ensure dispatch lands
    # on the Hetero builder rather than a homogeneous-method specialized struct.
    itp_otf = interp((x, y), data;
        method=(PchipInterp(), LinearInterp()),
        coeffs=OnTheFly(),
    )
    @test !hasfield(typeof(itp_otf), :spacings)
    # Was 9 (Tg,Tv,N,G,S,M,E,P,D), now 8 (drops S)
    @test length(typeof(itp_otf).parameters) == 8

    # Sanity: callable with finite Float64 result
    val = itp_otf((1.5, 1.5))
    @test val isa Float64
    @test isfinite(val)
end

