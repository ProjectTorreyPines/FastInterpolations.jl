# ============================================================================
# Graceful error messages for Hermite family ND
# ============================================================================
#
# PCHIP / Cardinal / Akima do not yet have a PreCompute backend, ND integrate,
# or ND adjoint. Rather than letting users hit MethodError or a confusing
# "use PreCompute" advice that dead-ends at another error, these entry points
# throw explicit ArgumentError messages pointing at the tracking TODOs and
# listing concrete workarounds.
#
# This test suite pins the error surface so the release contract is explicit.

@testitem "Hermite ND — graceful not-implemented errors" begin
    x = collect(range(0.0, 1.0, 8))
    y = collect(range(0.0, 1.0, 6))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]

    @testset "nodal_partials — pure local Hermite ND" begin
        # Pure Hermite tuple → OnTheFly is the only working coeffs strategy.
        # nodal_partials must tell the user this is not yet implemented and
        # point at the tracking TODO, NOT tell them to "use PreCompute" (which
        # would be rejected upstream).
        for m in (PchipInterp(), CardinalInterp(), AkimaInterp())
            itp = interp((x, y), data; method = (m, m))
            err = try
                nodal_partials(itp, (0, 0))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("not yet implemented for Hermite family ND", err.msg)
            @test occursin("hermite_nd_precompute.md", err.msg)
            # Must not give the misleading "use PreCompute" advice.
            @test !occursin("coeffs=PreCompute()", err.msg)
        end
    end

    @testset "nodal_partials — mixed Cubic × Hermite ND" begin
        # Mixed tuple also can't produce nodal_partials: PreCompute is rejected,
        # and OnTheFly has no partials array at all. Same error path.
        itp = interp((x, y), data; method = (CubicInterp(), PchipInterp()))
        err = try
            nodal_partials(itp, (1, 0))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Hermite family ND", err.msg)
        @test occursin("PchipInterp", err.msg)
    end

    @testset "nodal_partials — pure Cubic ND still works" begin
        # Regression guard: graceful-error path must NOT affect legitimate
        # Cubic ND callers.
        itp = interp((x, y), data; method = (CubicInterp(), CubicInterp()))
        @test size(nodal_partials(itp, (0, 0))) == size(data)
        @test size(nodal_partials(itp, (1, 0))) == size(data)
        @test size(nodal_partials(itp, (1, 1))) == size(data)
    end

    @testset "integrate(HeteroInterpolantND) — full-domain" begin
        itp = interp((x, y), data; method = (PchipInterp(), CardinalInterp()))
        err = try
            integrate(itp)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("not yet implemented for HeteroInterpolantND", err.msg)
        @test occursin("hermite_onthefly_integrate_and_nd_adjoint.md", err.msg)
        @test occursin("cubic_interp", err.msg)  # suggests homogeneous workaround
    end

    @testset "integrate(HeteroInterpolantND, a, b) — bounded" begin
        itp = interp((x, y), data; method = (CubicInterp(), AkimaInterp()))
        err = try
            integrate(itp, 0.1, 0.9)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("not yet implemented for HeteroInterpolantND", err.msg)
    end

    @testset "integrate(pure Cubic ND) — regression guard" begin
        # The HeteroInterpolantND override must NOT shadow the existing
        # CubicInterpolantND integrate path.
        itp = interp((x, y), data; method = (CubicInterp(), CubicInterp()))
        val = integrate(itp)
        @test val isa Real
        @test isfinite(val)
    end

    @testset "hetero_adjoint — rejects Hermite family" begin
        xq = [0.3, 0.7]
        yq = [0.4, 0.6]
        for m in (PchipInterp(), CardinalInterp(), AkimaInterp())
            err = try
                hetero_adjoint((x, y), (xq, yq); methods = (CubicInterp(), m))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("not yet implemented for", err.msg)
            @test occursin("Hermite family", err.msg)
            @test occursin("Workarounds", err.msg)
            @test occursin("ForwardDiff", err.msg)  # suggested workaround
        end
    end

    @testset "hetero_adjoint — pure Cubic×Linear still works" begin
        xq = [0.3, 0.7]
        yq = [0.4, 0.6]
        adj = hetero_adjoint((x, y), (xq, yq); methods = (CubicInterp(), LinearInterp()))
        y_bar = [1.0, 1.0]
        f_bar = adj(y_bar)
        @test size(f_bar) == size(data)
    end
end
