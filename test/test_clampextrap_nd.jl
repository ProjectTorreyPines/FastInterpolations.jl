# ClampExtrap ND forward-evaluation contract.
#
# ClampExtrap is a *flat* extension that clamps each coordinate independently:
#     f_ext(x, y) = f(clamp(x, loₓ, hiₓ), clamp(y, lo_y, hi_y))
# For a query OOB along x but in-domain along y, f_ext(x, y) = f(loₓ, y):
#   - it is CONSTANT in x          ⇒ ∂f/∂x = 0       (the OOB axis collapses)
#   - it still VARIES in y          ⇒ ∂f/∂y ≠ 0       (the in-domain axis keeps its slope)
# So the rule is axis-specific: a partial is zero iff it differentiates along an
# OOB axis; partials along in-domain axes survive. A blanket "OOB ⇒ zero deriv"
# (the 1D intuition) would wrongly zero the in-domain slope too.
#
# The ND *forward* path currently differentiates the clamped boundary cell, so the
# pure derivative along an OOB axis returns the boundary slope instead of 0. The
# in-domain-axis slopes are already correct. The defect is pinned with `@test_broken`.

@testitem "ClampExtrap ND forward contract" begin
    gx = 1:5
    gy = 1:5
    A = [Float64(i + j) for i in gx, j in gy]   # f(x,y) = x + y, ∂x ≡ ∂y ≡ 1 in-domain
    itp = linear_interp((gx, gy), A; extrap = ClampExtrap())

    @testset "value at OOB axis (clamped)" begin
        @test itp((0.0, 3.0)) ≈ 4.0    # x OOB-left → clamp x→1, f(1,3) = 4
        @test itp((0.0, 0.0)) ≈ 2.0    # both OOB → clamp to (1,1), f(1,1) = 2
        @test itp((2.5, 3.0)) ≈ 5.5    # in-domain, unaffected
    end

    @testset "in-domain-axis slope preserved under OOB on another axis" begin
        # Clamping one axis must NOT zero the derivative along the other axis.
        @test itp((0.0, 3.0); deriv = DerivOp(0, 1)) ≈ 1.0   # x OOB, ∂y still 1
        @test itp((3.0, 0.0); deriv = DerivOp(1, 0)) ≈ 1.0   # y OOB, ∂x still 1
        @test itp((2.5, 3.0); deriv = DerivOp(0, 1)) ≈ 1.0   # in-domain, ∂y = 1
    end

    @testset "deriv along an OOB axis must collapse to zero (KNOWN BUG)" begin
        # 1D reference: the OOB-axis derivative is correctly 0.
        i1 = linear_interp(gx, Float64.(collect(gx)); extrap = ClampExtrap())
        @test i1(0.0; deriv = DerivOp(1)) == 0.0

        # ND: pure derivative ALONG the OOB axis should be 0 (flat), currently 1.0.
        @test_broken itp((0.0, 3.0); deriv = DerivOp(1, 0)) == 0.0   # x OOB → ∂x
        @test_broken itp((3.0, 0.0); deriv = DerivOp(0, 1)) == 0.0   # y OOB → ∂y
    end
end
