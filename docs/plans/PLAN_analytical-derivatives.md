# Implementation Plan: Analytical Derivatives for FastInterpolations.jl

**Feature**: Add 1st and 2nd order analytical derivatives to linear and cubic interpolation
**Design Document**: [design/analytical_derivatives.md](../../design/analytical_derivatives.md)
**Created**: 2025-12-26
**Last Updated**: 2025-12-26
**Status**: Phase 2 Complete

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
- [ ] Add tests for `_eval_cubic_at_point` with op
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
- [ ] Update `src/cubic_eval.jl`
  - [ ] Add wrapper: `_eval_cubic_at_point(x,y,h,z,xi) = _eval_cubic_at_point(x,y,h,z,xi,EvalValue())`
  - [ ] Add full impl: `_eval_cubic_at_point(..., op::O) where {O<:AbstractEvalOp}`
  - [ ] Replace inline math with `_cubic_kernel(op, ...)` call
  - [ ] Same pattern for `_eval_cubic_at_point_periodic`
  - [ ] Add `_constant_extrap_result(op, y_boundary)` helper
  - [ ] Update `_eval_cubic_with_extrap` functions to accept and use op
  - [ ] Add wrappers for `_eval_with_bc` and `_cubic_vector_loop!`

#### REFACTOR: Clean up
- [ ] Verify all existing tests still pass (backward compat)
- [ ] Ensure consistent type parameter pattern

### Quality Gate
- [ ] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'`
- [ ] No type instability warnings (`@code_warntype`)
- [ ] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (no regressions)

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
- [ ] Add polynomial exactness tests (from design doc Section 7.1)
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
- [ ] Add allocation tests
- [ ] Add extrapolation derivative tests

#### GREEN: Implement to make tests pass
- [ ] Update `src/cubic_interp.jl`
  - [ ] Add `order::Int=0` to all public `cubic_interp` signatures
  - [ ] Add `@_dispatch_order order op begin ... end` wrapper
  - [ ] Create internal `_cubic_interp_impl` that takes `op::O`
- [ ] Update `src/cubic_interpolant.jl`
  - [ ] Add `derivative(itp::CubicInterpolant, xi)` scalar version
  - [ ] Add `derivative(itp::CubicInterpolant, x_query::AbstractVector)` vector version
  - [ ] Add `derivative2(itp::CubicInterpolant, xi)` scalar version
  - [ ] Add `derivative2(itp::CubicInterpolant, x_query::AbstractVector)` vector version
- [ ] Update exports in `src/FastInterpolations.jl`
  - [ ] Export `derivative`, `derivative2`
  - [ ] Export `cubic_derivative`, `cubic_derivative2` convenience functions

#### REFACTOR: Clean up
- [ ] Add docstrings to new public functions
- [ ] Verify backward compatibility (no order arg = value)

### Quality Gate
- [ ] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'`
- [ ] `@allocated cubic_interp(cache, y, 0.5; order=1) <= ALLOC_THRESHOLD`
- [ ] `@allocated cubic_interp(cache, y, 0.5; order=2) <= ALLOC_THRESHOLD`
- [ ] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (no regressions)

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
- [ ] Add linear derivative tests
  ```julia
  @testset "Linear derivatives" begin
      x = [0.0, 1.0, 3.0]
      y = [0.0, 2.0, 4.0]  # slopes: 2.0, 1.0

      @test linear_interp(x, y, 0.5; order=1) ≈ 2.0  # first segment
      @test linear_interp(x, y, 2.0; order=1) ≈ 1.0  # second segment
      @test linear_interp(x, y, 0.5; order=2) ≈ 0.0  # always zero
  end
  ```
- [ ] Add extrapolation mode tests for linear

#### GREEN: Implement to make tests pass
- [ ] Update `src/linear_interp.jl`
  - [ ] Add `_linear_eval_at_point(x, y, xi, op)` using `_linear_kernel`
  - [ ] Add `_linear_with_extrap(x, y, xi, ev, op)` for each extrap mode
    - [ ] `:none` - throws DomainError
    - [ ] `:constant` - returns 0 for derivatives outside
    - [ ] `:extension` - uses boundary interval
    - [ ] `:wrap` - wraps coordinate
  - [ ] Add `order::Int=0` to `linear_interp` and `linear_interp!`
  - [ ] Add `@_dispatch_order` wrapper
  - [ ] Add `derivative(itp::LinearInterpolant, xi)` scalar
  - [ ] Add `derivative(itp::LinearInterpolant, x_query::AbstractVector)` vector
  - [ ] Add `derivative2` functions (returns zero)
- [ ] Update exports
  - [ ] Add `linear_derivative` convenience function

#### REFACTOR: Clean up
- [ ] Add docstrings
- [ ] Verify all extrap modes work correctly

### Quality Gate
- [ ] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'`
- [ ] Extrapolation edge cases work correctly
- [ ] `@allocated linear_interp(x, y, 0.5; order=1) <= ALLOC_THRESHOLD`
- [ ] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (no regressions)

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
- [ ] Add Periodic BC derivative continuity tests (from design doc Section 7.4)
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
- [ ] Add boundary point tests (right-continuous behavior)
- [ ] Add type stability tests using `@inferred`
- [ ] Add comprehensive allocation tests for all paths

#### Documentation
- [ ] Update module docstring in `src/FastInterpolations.jl`
- [ ] Add examples to function docstrings
- [ ] Update README if exists

#### Final Validation
- [ ] Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
- [ ] Verify no regressions in existing tests
- [ ] Check allocation targets met

### Quality Gate
- [ ] Derivative tests pass: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_derivatives.jl"])'`
- [ ] Type stability verified for all public APIs (`@inferred`)
- [ ] Allocation thresholds met for all derivative orders
- [ ] `julia --project -e 'using Pkg; Pkg.test()'` - full test suite passes (no warnings)

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
| 3. Cubic Wrappers | Pending | - | - |
| 4. Cubic Public API | Pending | - | - |
| 5. Linear API | Pending | - | - |
| 6. Testing & Polish | Pending | - | - |

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
