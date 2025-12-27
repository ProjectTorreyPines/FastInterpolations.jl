# Implementation Plan: Analytical Derivatives for FastInterpolations.jl

**Feature**: Add 1st and 2nd order analytical derivatives to linear and cubic interpolation
**Design Document**: [design/analytical_derivatives.md](../../design/analytical_derivatives.md)
**Created**: 2025-12-26
**Last Updated**: 2025-12-27
**Status**: Phase 6 Complete (ALL PHASES DONE)

---

**CRITICAL INSTRUCTIONS**: After completing each phase:
1. Check off completed task checkboxes
2. Run all quality gate validation commands
3. Verify ALL quality gate items pass
4. Update "Last Updated" date
5. Document learnings in Notes section
6. Only then proceed to next phase

DO NOT skip quality gates or proceed with failing checks

---

## Overview

Add `order` parameter (0=value, 1=derivative, 2=second derivative) to all public APIs while maintaining:
- **Zero-allocation** for scalar queries
- **Type stability** throughout the call chain
- **Backward compatibility** via wrapper pattern

### Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dispatch strategy | `@_dispatch_order` at public API only | Single compile-time branch, no nested dispatch |
| Type parameter | `op::O where {O<:AbstractEvalOp}` | Ensures type stability (not `op::AbstractEvalOp`) |
| Backward compat | Wrapper pattern | Existing calls work unchanged |
| Kernel organization | Separate files | Independent testing, future SIMD/GPU optimization |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Type instability | Medium | High | Test with `@code_warntype`, enforce type parameter pattern |
| Allocation regression | Medium | High | Track with `@allocated`, use `ALLOC_THRESHOLD` |
| Periodic BC edge cases | Low | Medium | Test boundary continuity with small epsilon |
| Wrapper overhead | Low | Low | Inline annotations, verify with `@code_llvm` |

---

## Phase 1: Foundation (Types & Macro)

**Goal**: Establish core infrastructure - operation types, dispatch macro, and module structure

### Test Strategy
- **Test File**: `test/test_derivatives.jl` (new)
- **Coverage Target**: 100% of new ops.jl types
- **Test Scenarios**: Type instantiation, macro expansion, argument error

### Tasks

#### RED: Write failing tests first
- [x] Create `test/test_derivatives.jl` with basic type tests
  ```julia
  @testset "EvalOp types" begin
      @test EvalValue() isa AbstractEvalOp
      @test EvalDeriv1() isa AbstractEvalOp
      @test EvalDeriv2() isa AbstractEvalOp
  end
  ```
- [x] Add `@_dispatch_order` macro test
  ```julia
  @testset "@_dispatch_order macro" begin
      result = @_dispatch_order 0 op begin typeof(op) end
      @test result === EvalValue
      result = @_dispatch_order 1 op begin typeof(op) end
      @test result === EvalDeriv1
      @test_throws ArgumentError @_dispatch_order 3 op begin nothing end
  end
  ```

#### GREEN: Implement to make tests pass
- [x] Create `src/ops.jl`
  - [x] Define `AbstractEvalOp` abstract type
  - [x] Define `EvalValue`, `EvalDeriv1`, `EvalDeriv2` singleton structs
  - [x] Move `ExtrapVal` from cubic_types.jl to ops.jl
- [x] Update `src/utils.jl`
  - [x] Add `@_dispatch_order` macro
- [x] Update `src/FastInterpolations.jl`
  - [x] Add `include("ops.jl")` as first include (after imports)
  - [x] Export `AbstractEvalOp`, `EvalValue`, `EvalDeriv1`, `EvalDeriv2`

#### REFACTOR: Clean up
- [x] Ensure ops.jl has proper docstrings
- [x] Verify no duplicate definitions

### Quality Gate
- [x] `julia --project -e 'using FastInterpolations'` succeeds
- [x] `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'` passes (21 tests)
- [x] Types are documented with docstrings
- [x] Add `include("test_derivatives.jl")` to `test/runtests.jl`
- [x] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (1070 tests)

### Rollback
```bash
git checkout src/FastInterpolations.jl src/utils.jl test/runtests.jl
rm src/ops.jl test/test_derivatives.jl
```

---

## Phase 2: Kernel Files

**Goal**: Create pure mathematical kernel functions for linear and cubic evaluation

### Test Strategy
- **Test File**: `test/test_derivatives.jl` (extend)
- **Coverage Target**: 100% of kernel functions
- **Test Scenarios**: Known polynomial values, derivative correctness

### Tasks

#### RED: Write failing tests first
- [x] Add linear kernel tests
  ```julia
  @testset "Linear kernels" begin
      # L(x) = 1 + 2x on [0, 1], h=1, y0=1, y1=3
      h, y0, y1 = 1.0, 1.0, 3.0
      dt1 = 0.5  # x = 0.5
      @test _linear_kernel(EvalValue(), y0, y1, h, dt1) ≈ 2.0
      @test _linear_kernel(EvalDeriv1(), y0, y1, h, dt1) ≈ 2.0
      @test _linear_kernel(EvalDeriv2(), y0, y1, h, dt1) ≈ 0.0
  end
  ```
- [x] Add cubic kernel tests with known polynomial (quadratic & cubic exactness)

#### GREEN: Implement to make tests pass
- [x] Create `src/linear_kernels.jl`
  - [x] `_linear_kernel(::EvalValue, y0, y1, h, dt1)` - linear interpolation
  - [x] `_linear_kernel(::EvalDeriv1, y0, y1, h, dt1)` - constant slope
  - [x] `_linear_kernel(::EvalDeriv2, y0, y1, h, dt1)` - always zero
- [x] Create `src/cubic_kernels.jl`
  - [x] `_cubic_kernel(::EvalValue, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)`
  - [x] `_cubic_kernel(::EvalDeriv1, ...)` - first derivative formula
  - [x] `_cubic_kernel(::EvalDeriv2, ...)` - linear interpolation of z
- [x] Update `src/FastInterpolations.jl`
  - [x] Add `include("linear_kernels.jl")` after ops.jl
  - [x] Add `include("cubic_kernels.jl")` after linear_kernels.jl

#### REFACTOR: Clean up
- [x] Add `@inline` annotations to all kernel functions
- [x] Verify type parameter pattern used throughout

### Quality Gate
- [x] All kernel tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'` (53 tests)
- [x] `@code_warntype _linear_kernel(EvalDeriv1(), 1.0, 2.0, 1.0, 0.5)` shows no red (type stable)
- [x] `@code_warntype _cubic_kernel(EvalDeriv1(), 0.0, 0.0, 1.0, 2.0, 1.0, 0.5, 0.5)` shows no red (type stable)
- [x] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (1102 tests)

### Rollback
```bash
git checkout src/FastInterpolations.jl
rm src/linear_kernels.jl src/cubic_kernels.jl
```

---

## Phase 3: Cubic Internal Wrappers

**Goal**: Add op parameter to cubic evaluation functions with backward compatibility

### Test Strategy
- **Test File**: `test/test_derivatives.jl` (extend)
- **Coverage Target**: All cubic internal functions with op parameter
- **Test Scenarios**: Existing behavior preserved, new op parameter works

### Tasks

#### RED: Write failing tests first
- [x] Add tests for `_eval_cubic_at_point` with op
  ```julia
  @testset "Cubic internal functions with op" begin
      x = collect(0.0:0.5:2.0)
      y = x.^2
      cache = get_cubic_cache(x, Val(:natural))
      z = _solve_system!(cache, y, cache.bc_data)

      # Value at midpoint
      val = _eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalValue())
      @test val ≈ 1.0 atol=1e-10

      # Derivative at midpoint (f'(x) = 2x)
      deriv = _eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalDeriv1())
      @test deriv ≈ 2.0 atol=0.1  # Spline approximation
  end
  ```

#### GREEN: Implement to make tests pass
- [x] Update `src/cubic_eval.jl`
  - [x] Add wrapper: `_eval_cubic_at_point(x,y,h,z,xi) = _eval_cubic_at_point(x,y,h,z,xi,EvalValue())`
  - [x] Add full impl: `_eval_cubic_at_point(..., op::O) where {O<:AbstractEvalOp}`
  - [x] Replace inline math with `_cubic_kernel(op, ...)` call
  - [x] Same pattern for `_eval_cubic_at_point_periodic`
  - [x] Add `_constant_extrap_result(op, y_boundary)` helper
  - [x] Update `_eval_cubic_with_extrap` functions to accept and use op
  - [x] Add wrappers for `_eval_with_bc` and `_cubic_vector_loop!`

#### REFACTOR: Clean up
- [x] Verify all existing tests still pass (backward compat)
- [x] Ensure consistent type parameter pattern

### Quality Gate
- [x] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'` (69 tests)
- [x] No type instability warnings (`@code_warntype`)
- [x] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (1118 tests, no regressions)

### Rollback
```bash
git checkout src/cubic_eval.jl
```

---

## Phase 4: Cubic Public API

**Goal**: Add `order` parameter to all cubic public functions

### Test Strategy
- **Test File**: `test/test_derivatives.jl` (extend)
- **Coverage Target**: All cubic public API variants
- **Test Scenarios**: Polynomial exactness, allocation, extrapolation modes

### Tasks

#### RED: Write failing tests first
- [x] Add polynomial exactness tests (from design doc Section 7.1)
  ```julia
  @testset "Cubic polynomial exactness" begin
      x = collect(0.0:0.1:1.0)
      y = x.^2
      bc = BCPair(D2(2.0), D2(2.0))  # f''(x) = 2
      xi = 0.5
      @test cubic_interp(x, y, xi; bc=bc, order=0) ≈ 0.25 atol=1e-10
      @test cubic_interp(x, y, xi; bc=bc, order=1) ≈ 1.0 atol=1e-10
      @test cubic_interp(x, y, xi; bc=bc, order=2) ≈ 2.0 atol=1e-10
  end
  ```
- [x] Add allocation tests
- [x] Add extrapolation derivative tests

#### GREEN: Implement to make tests pass
- [x] Update `src/cubic_interp.jl`
  - [x] Add `order::Int=0` to all public `cubic_interp` signatures
  - [x] Add `@_dispatch_order order op begin ... end` wrapper
  - [x] Pass op through core implementations and vector loops
- [x] Update `src/cubic_interpolant.jl`
  - [x] Add `derivative(itp::CubicInterpolant, xi)` scalar version
  - [x] Add `derivative(itp::CubicInterpolant, x_query::AbstractVector)` vector version
  - [x] Add `derivative2(itp::CubicInterpolant, xi)` scalar version
  - [x] Add `derivative2(itp::CubicInterpolant, x_query::AbstractVector)` vector version
- [x] Update exports in `src/FastInterpolations.jl`
  - [x] Export `derivative`, `derivative2`
- [x] Update `src/cubic_eval.jl`
  - [x] Add op parameter to `_eval_with_bc` functions
  - [x] Add op parameter to `_cubic_vector_loop!` functions
  - [x] Add op parameter to `cubic_interp_scalar`

#### REFACTOR: Clean up
- [x] Add docstrings to new public functions
- [x] Verify backward compatibility (no order arg = value)

### Quality Gate
- [x] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'` (107 tests)
- [x] `@allocated cubic_interp(cache, y, 0.5; order=1) == 0`
- [x] `@allocated cubic_interp(cache, y, 0.5; order=2) == 0`
- [x] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (1156 tests, no regressions)

### Rollback
```bash
git checkout src/cubic_interp.jl src/cubic_interpolant.jl src/FastInterpolations.jl
```

---

## Phase 5: Linear API

**Goal**: Add `order` parameter to linear interpolation with all extrapolation modes

### Test Strategy
- **Test File**: `test/test_derivatives.jl` (extend)
- **Coverage Target**: All linear public API and internal functions
- **Test Scenarios**: Constant slope, zero 2nd derivative, extrapolation modes

### Tasks

#### RED: Write failing tests first
- [x] Add linear derivative tests
  ```julia
  @testset "Linear derivatives" begin
      x = [0.0, 1.0, 3.0]
      y = [0.0, 2.0, 4.0]  # slopes: 2.0, 1.0

      @test linear_interp(x, y, 0.5; order=1) ≈ 2.0  # first segment
      @test linear_interp(x, y, 2.0; order=1) ≈ 1.0  # second segment
      @test linear_interp(x, y, 0.5; order=2) ≈ 0.0  # always zero
  end
  ```
- [x] Add extrapolation mode tests for linear

#### GREEN: Implement to make tests pass
- [x] Update `src/linear_interp.jl`
  - [x] Add `_linear_eval_at_point(x, y, xi, op)` using `_linear_kernel`
  - [x] Add `_linear_with_extrap(x, y, xi, ev, op)` for each extrap mode
    - [x] `:none` - throws DomainError
    - [x] `:constant` - returns 0 for derivatives outside
    - [x] `:extension` - uses boundary interval
    - [x] `:wrap` - wraps coordinate
  - [x] Add `order::Int=0` to `linear_interp` and `linear_interp!`
  - [x] Add `@_dispatch_order` wrapper
  - [x] Add `derivative(itp::LinearInterpolant, xi)` scalar
  - [x] Add `derivative(itp::LinearInterpolant, x_query::AbstractVector)` vector
  - [x] Add `derivative2` functions (returns zero)
- [x] Update exports
  - [x] `derivative`/`derivative2` already exported (shared with CubicInterpolant)

#### REFACTOR: Clean up
- [x] Add docstrings
- [x] Verify all extrap modes work correctly
- [x] Fix `LinearInterpolant.mode` type from `Val` to `ExtrapVal` for zero-allocation

### Quality Gate
- [x] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'` (247 tests)
- [x] Extrapolation edge cases work correctly
- [x] `@allocated linear_interp(x, y, 0.5; order=1) <= ALLOC_THRESHOLD`
- [x] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (1293 tests, no regressions)

### Rollback
```bash
git checkout src/linear_interp.jl src/FastInterpolations.jl
```

---

## Phase 6: Comprehensive Testing & Polish

**Goal**: Full test coverage, edge cases, and documentation

### Test Strategy
- **Test File**: `test/test_derivatives.jl` (finalize)
- **Coverage Target**: 90%+ for all derivative-related code
- **Test Scenarios**: Periodic BC, boundary points, type stability

### Tasks

#### Testing
- [x] Add Periodic BC derivative continuity tests (from design doc Section 7.4)
  ```julia
  @testset "Periodic BC derivative continuity" begin
      x = range(0, 2π, 101) |> collect
      y = sin.(x)
      y[end] = y[1]
      ε = 1e-6
      d_left = cubic_interp(x, y, 2π - ε; bc=PeriodicBC(), order=1)
      d_right = cubic_interp(x, y, ε; bc=PeriodicBC(), order=1)
      @test d_left ≈ d_right atol=1e-4
  end
  ```
- [x] Add boundary point tests (right-continuous behavior)
- [x] Add type stability tests using `@inferred`
- [x] Add comprehensive allocation tests for all paths

#### Documentation
- [x] Docstrings added to all derivative functions in Phase 4-5
- [x] Examples included in docstrings
- [x] Module exports documented

#### Final Validation
- [x] Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
- [x] Verify no regressions in existing tests
- [x] Check allocation targets met

### Quality Gate
- [x] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'` (394 tests)
- [x] Type stability verified for all public APIs (`@inferred`)
- [x] Allocation thresholds met for all derivative orders
- [x] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (1440 tests)

### Rollback
```bash
git checkout .
```

---

## Progress Tracking

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| 1. Foundation | ✅ Complete | 2025-12-26 | 2025-12-26 |
| 2. Kernel Files | ✅ Complete | 2025-12-26 | 2025-12-26 |
| 3. Cubic Wrappers | ✅ Complete | 2025-12-26 | 2025-12-26 |
| 4. Cubic Public API | ✅ Complete | 2025-12-27 | 2025-12-27 |
| 5. Linear API | ✅ Complete | 2025-12-27 | 2025-12-27 |
| 6. Testing & Polish | ✅ Complete | 2025-12-27 | 2025-12-27 |

---

## Notes & Learnings

### Phase 1 (2025-12-26)
- Foundation was pre-implemented during design phase:
  - `src/ops.jl`: Types (`AbstractEvalOp`, `EvalValue`, `EvalDeriv1`, `EvalDeriv2`, `ExtrapVal`)
  - `src/utils.jl`: `@_dispatch_order` macro
  - `src/FastInterpolations.jl`: Includes and exports
- Added comprehensive test coverage in `test/test_derivatives.jl`:
  - Type instantiation tests
  - Macro dispatch tests (order 0/1/2)
  - Runtime variable dispatch tests
  - Type stability verification with `@inferred`
- All 21 tests pass

### Phase 2 (2025-12-26)
- Kernel files were pre-implemented during design phase:
  - `src/linear_kernels.jl`: `_linear_kernel` for Value/Deriv1/Deriv2
  - `src/cubic_kernels.jl`: `_cubic_kernel` for Value/Deriv1/Deriv2
- Added comprehensive kernel tests:
  - Linear: value interpolation, constant slope, zero 2nd derivative
  - Cubic: quadratic exactness (f(x)=x²), cubic exactness (f(x)=x³)
  - Type stability tests with `@inferred`
  - Float32 preservation tests
- All kernels have `@inline` annotations
- 53 total derivative tests pass (21 Phase 1 + 32 Phase 2)

### Phase 3 (2025-12-26)
- Implemented backward-compatible wrappers in `src/cubic_eval.jl`:
  - `_eval_cubic_at_point` with 5 and 6-arg forms (op defaults to EvalValue())
  - `_eval_cubic_at_point_periodic` with 6 and 7-arg forms
  - All `_eval_cubic_with_extrap` variants for :none, :constant, :extension, :wrap
  - Added `_constant_extrap_result(op, y_boundary)` helper for constant extrapolation
- All internal functions now use `_cubic_kernel(op, ...)` for evaluation
- Pattern: wrapper → full impl with `op::O where {O<:AbstractEvalOp}` → kernel
- Added 16 Phase 3 tests:
  - `_eval_cubic_at_point` with op (value, deriv1, deriv2)
  - Backward compatibility verification
  - `_eval_cubic_with_extrap` with constant/extension modes
  - Type stability tests with `@inferred`
  - Derivative at multiple points
- 69 total derivative tests pass (21 Phase 1 + 32 Phase 2 + 16 Phase 3)
- Full test suite: 1118 tests pass with no regressions

### Phase 4 (2025-12-27)
- Added `order::Int=0` parameter to all public `cubic_interp` signatures:
  - `cubic_interp(cache, y, x_query; extrap, order)`
  - `cubic_interp(x, y, x_query; bc, extrap, autocache, order)`
  - All in-place variants `cubic_interp!` as well
  - Generic Real type wrappers updated
- Updated internal functions to pass op parameter through call chain:
  - `_cubic_interp_bcpair!`, `_cubic_interp_bcpair_scalar`
  - `_cubic_interp_periodic!`, `_cubic_interp_periodic_scalar`
  - `_eval_with_bc` (Periodic and BCPair variants)
  - `_cubic_vector_loop!` (default and Periodic variants)
  - `cubic_interp_scalar`
- Added `derivative` and `derivative2` functions for CubicInterpolant:
  - Scalar versions (zero-allocation)
  - Vector versions
  - Both forward to `_eval_with_bc` with appropriate EvalOp
- Exported `derivative`, `derivative2` in FastInterpolations.jl
- Added 38 Phase 4 tests:
  - Polynomial exactness (quadratic and cubic)
  - Backward compatibility verification
  - Vector query with order
  - Cache-based with order
  - Type stability with `@inferred`
  - Zero-allocation verification
  - Extrapolation derivatives (constant/extension)
  - CubicInterpolant derivative methods
- 107 total derivative tests pass (69 Phase 1-3 + 38 Phase 4)
- Full test suite: 1156 tests pass with no regressions

### Phase 5 (2025-12-27)
- Added `order::Int=0` parameter to all public `linear_interp` signatures:
  - `linear_interp(x, y, xi; extrap, order)`
  - `linear_interp(x, y, x_targets; extrap, order)`
  - All in-place variants `linear_interp!` as well
  - Range-optimized variants updated
- Added internal evaluation functions with op parameter:
  - `_linear_eval_at_point(x, y, xi, extrap, op)` using `_linear_kernel`
  - `_linear_eval_constant_extrap(y, is_left, op)` for constant extrapolation handling
  - `_linear_with_extrap(x, y, xi, extrap, op)` for all extrapolation modes
- Added `derivative` and `derivative2` functions for LinearInterpolant:
  - Scalar versions (zero-allocation)
  - Vector versions
  - Both forward to `_linear_with_extrap` with appropriate EvalOp
- **Bug fix**: Changed `LinearInterpolant.mode` from `Val` to `ExtrapVal` (concrete union type)
  - The abstract `Val` type caused dynamic dispatch and allocation
  - Using `ExtrapVal = Union{Val{:none}, Val{:constant}, Val{:extension}, Val{:wrap}}` enables union-splitting
  - This matches the pattern used by `CubicInterpolant.extrap`
- Added 140 Phase 5 tests:
  - Constant slope segments verification
  - Backward compatibility verification
  - Vector query with order
  - In-place operations with order
  - Type stability with `@inferred`
  - Zero-allocation verification
  - All extrapolation modes (none, constant, extension, wrap)
  - LinearInterpolant derivative methods (scalar and vector)
  - Range optimization preservation
- 247 total derivative tests pass
- Full test suite: 1293 tests pass with no regressions

### Phase 6 (2025-12-27)
- Added comprehensive testing for Periodic BC derivative continuity:
  - First and second derivative continuity at wrap boundaries
  - Tested sin(x) and cos(x) periodic functions
  - Verified derivatives match analytical values (cos for sin', -sin for sin'')
- Added boundary point behavior tests:
  - Cubic at knot points (C1 continuity verified)
  - Linear at knot points (right-continuous behavior)
  - Derivative consistency across knots with ε-testing
- Added comprehensive type stability tests:
  - `@inferred` for all derivative functions (scalar/vector)
  - Float32 type preservation verified
  - Order parameter type inference verified
  - Cache-based type inference verified
- Added comprehensive allocation tests:
  - All cubic BC types × extrap modes × orders (45 combinations)
  - All linear extrap modes × orders (12 combinations)
  - Range path allocation verification
  - Cache-based allocation verification
- Added edge case and robustness tests:
  - Minimum grid size (2-3 points)
  - Domain boundary queries
  - Constant and linear functions
  - Non-uniform grids
  - Large grids (1001 points)
- 394 total derivative tests pass (247 Phase 1-5 + 147 Phase 6)
- Full test suite: 1440 tests pass with no regressions

---

## Quick Reference

### Key Files to Create
- `src/ops.jl` - Operation types
- `src/linear_kernels.jl` - Linear kernel functions
- `src/cubic_kernels.jl` - Cubic kernel functions

### Key Files to Modify
- `src/FastInterpolations.jl` - Module includes and exports
- `src/utils.jl` - Add `@_dispatch_order` macro
- `src/cubic_eval.jl` - Add op parameter with wrappers
- `src/cubic_interp.jl` - Add order parameter
- `src/cubic_interpolant.jl` - Add derivative functions
- `src/linear_interp.jl` - Add order parameter and derivative functions

### Test Commands
```bash
# Run only derivative tests
julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'

# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Check type stability
julia --project -e 'using FastInterpolations; @code_warntype cubic_interp([0.0,1.0], [0.0,1.0], 0.5; order=1)'

# Check allocation
julia --project -e 'using FastInterpolations; cache=get_cubic_cache([0.0,0.5,1.0], Val(:natural)); y=[0.0,0.25,1.0]; cubic_interp(cache,y,0.5;order=1); @allocated cubic_interp(cache,y,0.5;order=1)'
```
