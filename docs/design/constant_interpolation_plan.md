# Constant Interpolation Implementation Plan

## Overview
FastInterpolations.jl에 constant (step/piecewise constant) interpolation 추가

## Design Decisions (Confirmed)
- **Keyword**: `side=:nearest/:left/:right` (default: `:nearest`)
- **Tie-breaking**: Left (floor convention) - midpoint에서 왼쪽 값 사용
- **Callable form**: `ConstantInterpolant` 구현 (2-arg form)
- **Derivatives**: 1차, 2차 모두 0 반환 (constant 함수이므로)
- **Grid requirement**: `length(x) >= 2` (라이브러리 일관성 유지, `_find_interval` 전제조건)
- **Grid point behavior**: `xi == x[i]`일 때 모든 side 옵션에서 `y[i]` 반환 (수학적 좌/우연속 정의)
- **Boundary handling**: `xi == x[end]` 명시적 특수 처리 (`:wrap` 제외, 나머지 extrap 모드에서 `y[end]` 직접 반환)
- **Extrapolation + side**: 외삽 구간에서는 side 무시, 경계값 직접 반환 (`xi < x_min → y[1]`, `xi > x_max → y[end]`)
- **Type policy**: Float 강제 변환 (linear_interp와 동일, 내부 유틸이 AbstractFloat 전제)
- **Wrap domain**: `[x_min, x_max)` 반개구간, `xi == x_max`는 `x_min`으로 wrap
  - ⚠️ **라이브러리 일관성 확인**: `linear_interp`, `cubic_interp` 모두 동일한 `_wrap_to_domain` 사용
  - `utils.jl:111-116`: `xi >= x_max` → `x_min + mod(xi - x_min, period)` → `xi == x_max` → `x_min`
- **Grid point tolerance**: `iszero(dt1)` strict 비교 (허용오차 없음)

---

## Phase 1: Infrastructure

### 1.1 `src/ops.jl` - SideVal 타입 추가
```julia
const SideVal = Union{Val{:nearest}, Val{:left}, Val{:right}}
```

### 1.2 `src/utils.jl` - @_dispatch_side 매크로 추가
`@_dispatch_extrap` 패턴을 따라 side 심볼을 Val 타입으로 변환하는 매크로

```julia
macro _dispatch_side(pair, body)
    pair.head === :call && pair.args[1] === :(=>) ||
        error("@_dispatch_side expects `sym => varname`, got: $pair")
    sym = pair.args[2]
    varname = pair.args[3]
    svs = esc(varname)
    quote
        let _side = $(esc(sym))
            if _side === :nearest
                $svs = Val(:nearest)
                $(esc(body))
            elseif _side === :left
                $svs = Val(:left)
                $(esc(body))
            elseif _side === :right
                $svs = Val(:right)
                $(esc(body))
            else
                throw(ArgumentError("`side` must be :nearest, :left, or :right, got :$_side"))
            end
        end
    end
end
```

---

## Phase 2: Kernel Implementation

### 2.1 새 파일: `src/constant_kernels.jl`

**Signature**: `_constant_kernel(op::AbstractEvalOp, y_left, y_right, h, dt1, side::SideVal)`
- 기존 `_linear_kernel`, `_cubic_kernel`과 동일한 인터페이스 패턴
- `op` 파라미터가 첫 번째 (AbstractEvalOp dispatch)
- **Grid point 감지**: `iszero(dt1)` strict 비교 (허용오차 없음)

```julia
# EvalValue dispatch - side에 따라 값 선택
# 중요: dt1 == 0 (정확히 grid point)일 때는 모든 side에서 y_left (해당 point 값) 반환

@inline function _constant_kernel(::EvalValue, y_left::T, y_right::T, h::T, dt1::T, ::Val{:left}) where {T}
    return y_left  # 항상 왼쪽 값
end

@inline function _constant_kernel(::EvalValue, y_left::T, y_right::T, h::T, dt1::T, ::Val{:right}) where {T}
    # dt1 == 0 → grid point → y_left, 그 외 → y_right
    return iszero(dt1) ? y_left : y_right
end

@inline function _constant_kernel(::EvalValue, y_left::T, y_right::T, h::T, dt1::T, ::Val{:nearest}) where {T}
    # dt1 == 0 → grid point → y_left
    # dt1 <= h/2 → y_left (left tie-breaking 포함)
    return dt1 <= h / 2 ? y_left : y_right
end

# EvalDeriv1/EvalDeriv2 - 항상 zero(T) 반환 (타입 보존)
@inline function _constant_kernel(::EvalDeriv1, y_left::T, ::T, ::T, ::T, ::SideVal) where {T}
    return zero(T)
end

@inline function _constant_kernel(::EvalDeriv2, y_left::T, ::T, ::T, ::T, ::SideVal) where {T}
    return zero(T)
end
```

---

## Phase 3: Main API

### 3.1 새 파일: `src/constant_interp.jl`

**함수 시그니처:**
```julia
# Scalar
constant_interp(x, y, xi; extrap=:none, side=:nearest, deriv=0)

# Vector (in-place)
constant_interp!(output, x, y, x_targets; extrap=:none, side=:nearest, deriv=0)

# Vector (allocating)
constant_interp(x, y, x_targets; extrap=:none, side=:nearest, deriv=0)

# 2-arg callable form
constant_interp(x, y; extrap=:none, side=:nearest) → ConstantInterpolant
```

**ConstantInterpolant struct** (Union-splitting 최적화):
```julia
# 타입 파라미터에 S를 넣지 않고 SideVal Union 사용 → 타입 시그니처 단순화
# T는 AbstractFloat (linear_interp와 동일, 내부 유틸 호환)
struct ConstantInterpolant{T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
    x::X
    y::Y
    mode::ExtrapVal   # Union{Val{:none}, Val{:constant}, Val{:extension}, Val{:wrap}}
    side::SideVal     # Union{Val{:nearest}, Val{:left}, Val{:right}}
end
```

**ConstantInterpolant 호출 API** (LinearInterpolant 패턴):
```julia
# ─────────────────────────────────────────────────────────────
# Scalar call - hot path (inlined for broadcast fusion)
# ─────────────────────────────────────────────────────────────
@inline function (itp::ConstantInterpolant{T})(xi::T; deriv::Int=0) where {T}
    @_dispatch_deriv deriv => op begin
        _constant_eval_at_point(itp.x, itp.y, xi, itp.mode, itp.side, op)
    end
end

# Real scalar wrapper - delegates to T method
@inline function (itp::ConstantInterpolant{T})(xi::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    itp(T(xi); deriv=deriv)
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T})(xi::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    xi_typed = S === T ? xi : T.(xi)
    output = Vector{T}(undef, length(xi_typed))
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.mode, itp.side, op)
        end
    end
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call (zero allocation)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi, itp.mode)
        @inbounds for i in eachindex(xi, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi[i], itp.mode, itp.side, op)
        end
    end
    return output
end

# In-place with type conversion
function (itp::ConstantInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = T.(xi)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.mode, itp.side, op)
        end
    end
    return output
end
```

**Real 입력 래퍼** (linear_interp 패턴):

⚠️ **타입 정책**: `x`, `y`는 **동일 타입** 요구 (linear_interp와 동일)
- `x::AbstractVector{T}, y::AbstractVector{T}` (T<:Real)
- `xi`는 별도 타입 `S<:Real` 허용 (쿼리 포인트)

```julia
# ─────────────────────────────────────────────────────────────
# Scalar Real → Float 변환 래퍼
# ─────────────────────────────────────────────────────────────
function constant_interp(x::AbstractVector{T}, y::AbstractVector{T}, xi::S;
                         extrap=:none, side=:nearest, deriv=0) where {T<:Real, S<:Real}
    FT = float(T)
    return constant_interp(_to_float(x, FT), FT.(y), FT(xi); extrap, side, deriv)
end

# ─────────────────────────────────────────────────────────────
# Vector Real → Float 변환 래퍼
# ─────────────────────────────────────────────────────────────
function constant_interp(x::AbstractVector{T}, y::AbstractVector{T}, xi::AbstractVector{S};
                         extrap=:none, side=:nearest, deriv=0) where {T<:Real, S<:Real}
    FT = float(T)
    return constant_interp(_to_float(x, FT), FT.(y), FT.(xi); extrap, side, deriv)
end

# ─────────────────────────────────────────────────────────────
# In-place Real → Float 변환 래퍼
# ─────────────────────────────────────────────────────────────
function constant_interp!(output::AbstractVector, x::AbstractVector{T}, y::AbstractVector{T},
                          xi::AbstractVector{S}; extrap=:none, side=:nearest, deriv=0) where {T<:Real, S<:Real}
    FT = float(T)
    return constant_interp!(output, _to_float(x, FT), FT.(y), FT.(xi); extrap, side, deriv)
end

# ─────────────────────────────────────────────────────────────
# 2-arg Callable Real → Float 변환 래퍼
# ─────────────────────────────────────────────────────────────
function constant_interp(x::AbstractVector{T}, y::AbstractVector{T};
                         extrap=:none, side=:nearest) where {T<:Real}
    FT = float(T)
    return constant_interp(_to_float(x, FT), FT.(y); extrap, side)
end
```

**Note**: `_to_float(x, FT)` preserves Range structure for O(1) lookup, while `FT.(x)` converts to Vector.

**Performance notes:**
- `SideVal`과 `ExtrapVal` 모두 Union-splitting으로 최적화됨
- `_find_interval_with_bounds` 호출 시 `AbstractRange` → O(1), `AbstractVector` → O(log n) 자동 dispatch
- Real 입력은 Float으로 변환 (linear_interp와 동일)

**Boundary handling** (`xi == x[end]` 특수 처리, `:wrap` 제외):
```julia
# 평가 흐름: domain check → wrap 처리 → boundary check → extrap check → interval search → kernel
@inline function _constant_eval_at_point(x, y, xi, extrap, side, op)
    # ⚠️ :none 도메인 체크 - DomainError를 던짐 (linear_interp 패턴)
    # _check_domain은 :none일 때만 체크, 나머지는 no-op
    @boundscheck _check_domain(x, xi, extrap)

    x_min, x_max = first(x), last(x)

    # 0. :wrap 모드는 먼저 처리 (반개구간 [x_min, x_max) 의미)
    #    xi == x_max → x_min으로 wrap되어 side 규칙 적용
    if extrap === Val(:wrap)
        xi_wrapped = _wrap_to_domain(xi, x_min, x_max)
        # wrap 후에는 xi_wrapped ∈ [x_min, x_max)이므로 일반 경로로
        idx, x0, x1 = _find_interval_with_bounds(x, xi_wrapped)
        h = x1 - x0
        dt1 = xi_wrapped - x0
        @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dt1, side)
    end

    # 1. 명시적 boundary 처리 (xi == x[end], :wrap 제외)
    #    _find_interval_with_bounds가 idx=n-1, dt1=h를 반환하는 문제 우회
    if xi == x_max
        # deriv에 따라 다른 값 반환
        return op isa EvalValue ? (@inbounds y[end]) : zero(eltype(y))
    end

    # 2. 외삽 처리 (:constant, :extension만 - :none은 위에서 DomainError, :wrap은 0번에서 처리)
    if xi < x_min || xi > x_max
        return _constant_eval_extrap(y, xi, x_min, x_max, extrap, side, op)
    end

    # 3. 일반 interval 탐색 및 커널 호출
    idx, x0, x1 = _find_interval_with_bounds(x, xi)
    h = x1 - x0
    dt1 = xi - x0
    @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dt1, side)
end
```

**:none 도메인 체크 흐름:**
- `_check_domain(x, xi, Val(:none))` → `xi ∉ [x_min, x_max]`이면 `DomainError`
- `_check_domain(x, xi, Val(:constant/:extension/:wrap))` → no-op (utils.jl:175)
- 따라서 `_constant_eval_extrap`에는 `Val{:none}` 오버로드 불필요

**Extrapolation 동작** (side 무시, 경계값 직접 반환, **deriv=1/2는 0 반환**):
```julia
# EvalValue: 경계값 반환
@inline function _constant_eval_extrap(y::AbstractVector{T}, xi, x_min, x_max, ::Val{:constant}, ::SideVal, ::EvalValue) where {T}
    if xi < x_min
        return @inbounds y[1]
    else  # xi > x_max
        return @inbounds y[end]
    end
end

# EvalDeriv1/EvalDeriv2: 항상 0 반환 (constant 함수의 미분)
@inline function _constant_eval_extrap(y::AbstractVector{T}, ::Any, ::Any, ::Any, ::Val{:constant}, ::SideVal, ::EvalDeriv1) where {T}
    return zero(T)
end

@inline function _constant_eval_extrap(y::AbstractVector{T}, ::Any, ::Any, ::Any, ::Val{:constant}, ::SideVal, ::EvalDeriv2) where {T}
    return zero(T)
end

# :extension은 :constant와 동일 (slope=0이므로)
@inline function _constant_eval_extrap(y::AbstractVector{T}, xi, x_min, x_max, ::Val{:extension}, side::SideVal, op::AbstractEvalOp) where {T}
    return _constant_eval_extrap(y, xi, x_min, x_max, Val(:constant), side, op)
end

# :none은 이 함수 호출 전에 DomainError

# ⚠️ :wrap은 _constant_eval_extrap에 오버로드 없음!
# :wrap은 _constant_eval_at_point에서 먼저 처리됨 (line 189-196)
# xi가 도메인 밖이든 안이든, wrap 후 interval search → kernel 호출
```

**Extrapolation 모드별 동작:**
- `:none` → DomainError (도메인 밖 접근 시)
- `:constant` → `y[1]` 또는 `y[end]` (side 무시), **deriv=1/2는 `zero(T)` 반환**
- `:extension` → `:constant`와 동일 (slope이 없으므로)
- `:wrap` → **`_constant_eval_extrap` 호출 전에 처리됨** (line 189-196)
  - `_wrap_to_domain`으로 `[x_min, x_max)` 반개구간으로 wrap
  - wrap 후 interval search → kernel 호출 → side 규칙 적용
  - 외삽 분기(`_constant_eval_extrap`)를 거치지 않음

---

## Phase 4: Integration

### 4.1 `src/FastInterpolations.jl` 수정
```julia
# includes 추가 (linear_kernels.jl 뒤에)
include("constant_kernels.jl")
include("constant_interp.jl")

# exports 추가
export constant_interp, constant_interp!, ConstantInterpolant
```

### 4.2 `src/derivative_view.jl` 수정
```julia
deriv1(itp::ConstantInterpolant) = DerivativeView{1, typeof(itp)}(itp)
deriv2(itp::ConstantInterpolant) = DerivativeView{2, typeof(itp)}(itp)
```

---

## Phase 5: Tests

### 5.1 새 파일: `test/test_constant.jl`

**Test categories:**
1. Core functionality (uniform/non-uniform grid, all side options)
2. Extrapolation modes (:none, :constant, :extension, :wrap)
3. Derivatives (항상 0)
4. ConstantInterpolant callable (scalar, vector, broadcast, `deriv` keyword)
5. DerivativeView integration (`deriv1(itp)`, `deriv2(itp)`)
6. Zero-allocation verification (`@allocated` 사용, BenchmarkTools 불필요)
7. Type stability (`@inferred`)
8. Edge cases:
   - **Grid point behavior**: `xi == x[i]`에서 모든 side 옵션이 `y[i]` 반환
   - **Boundary special case**: `xi == x[end]`에서 `y[end]` 반환 (`:wrap` 제외, `:wrap`은 `x_min`으로 wrap)
   - Floating-point tolerance (midpoint 근처)
   - Grid size validation (`length(x) >= 2`)
   - **Extrapolation + side**: 외삽 시 side 무시 확인
   - **Wrap domain**: `xi == x_max` → `x_min`으로 wrap
9. Broadcast fusion 성능 (`@. coef * itp(rho)` 형태)
10. Real 입력 변환 (scalar, vector, in-place, 2-arg callable 모두 확인)

**추가 테스트 케이스 (리뷰 반영):**
```julia
# ─────────────────────────────────────────────────────────────
# Wrap + boundary 충돌 케이스
# ─────────────────────────────────────────────────────────────
@test constant_interp(x, y, x[end]; extrap=:wrap, side=:left) == y[1]   # x_max → x_min wrap
@test constant_interp(x, y, x[end]; extrap=:wrap, side=:right) == y[1]  # wrap 후 side 적용

# ─────────────────────────────────────────────────────────────
# Extrapolation + deriv 조합 (deriv=1/2는 항상 0)
# ─────────────────────────────────────────────────────────────
@test constant_interp(x, y, x[1] - 1.0; extrap=:constant, deriv=1) == 0.0
@test constant_interp(x, y, x[end] + 1.0; extrap=:constant, deriv=2) == 0.0
@test constant_interp(x, y, x[1] - 1.0; extrap=:extension, deriv=1) == 0.0
@test constant_interp(x, y, x[end] + 1.0; extrap=:wrap, deriv=1) == 0.0

# ─────────────────────────────────────────────────────────────
# :none + xi == x_max는 DomainError가 아님 (domain 내)
# ─────────────────────────────────────────────────────────────
@test constant_interp(x, y, x[end]; extrap=:none) == y[end]  # NOT DomainError

# ─────────────────────────────────────────────────────────────
# xi == x_min with side=:right returns y[1]
# ─────────────────────────────────────────────────────────────
@test constant_interp(x, y, x[1]; side=:right) == y[1]  # dt1 == 0 → y_left

# ─────────────────────────────────────────────────────────────
# Real 입력 래퍼 검증 (scalar, vector, in-place, 2-arg callable)
# ─────────────────────────────────────────────────────────────
x_int = [1, 2, 3, 4, 5]
y_int = [10, 20, 30, 40, 50]
@test constant_interp(x_int, y_int, 2.5) isa Float64           # scalar
@test constant_interp(x_int, y_int, [1.5, 2.5]) isa Vector{Float64}  # vector
itp_int = constant_interp(x_int, y_int)  # 2-arg callable
@test itp_int(2.5) isa Float64

# In-place with Real inputs
out = zeros(2)
constant_interp!(out, x_int, y_int, [1.5, 2.5])
@test out ≈ [10.0, 20.0]  # with side=:nearest default

# ─────────────────────────────────────────────────────────────
# NaN/Inf 동작 (문서화 목적, undefined behavior 허용)
# ─────────────────────────────────────────────────────────────
# Note: NaN/Inf 입력은 undefined behavior (caller's responsibility)
# 문서화만 하고 강제 테스트는 불필요
```

### 5.2 `test/runtests.jl` 수정
```julia
include("test_constant.jl")
```

---

## Files to Create
| File | Purpose |
|------|---------|
| `src/constant_kernels.jl` | Pure math kernel functions |
| `src/constant_interp.jl` | API, ConstantInterpolant struct |
| `test/test_constant.jl` | Test suite |

## Files to Modify
| File | Changes |
|------|---------|
| `src/ops.jl` | Add `SideVal` union type (~line 47) |
| `src/utils.jl` | Add `@_dispatch_side` macro (~line 345) |
| `src/FastInterpolations.jl` | Add includes and exports |
| `src/derivative_view.jl` | Add deriv1/deriv2 for ConstantInterpolant |
| `test/runtests.jl` | Include test_constant.jl |

---

## Implementation Order
1. `src/ops.jl` - SideVal 추가
2. `src/utils.jl` - @_dispatch_side 매크로
3. `src/constant_kernels.jl` - 커널 함수
4. `src/constant_interp.jl` - API 및 ConstantInterpolant
5. `src/FastInterpolations.jl` - includes/exports
6. `src/derivative_view.jl` - derivative view 지원
7. `test/test_constant.jl` - 테스트
8. `test/runtests.jl` - 테스트 포함

---

## Review Checklist (구현 시 확인사항)

| Item | Description |
|------|-------------|
| Union-splitting | `SideVal`, `ExtrapVal` 모두 concrete Union으로 정의되어 최적화 |
| O(1) lookup | `AbstractRange` 입력 시 `_find_interval_with_bounds` O(1) 경로 활용 |
| Float 변환 | Real 입력은 Float으로 변환 (linear_interp와 동일), Range 구조 보존 |
| ArgumentError | `@_dispatch_side`에서 잘못된 심볼 입력 시 명확한 에러 메시지 |
| Grid validation | `length(x) >= 2` 전제조건 문서화 (assertion 또는 docstring) |
| Broadcast fusion | `ConstantInterpolant`가 `@.` 구문에서 성능 저하 없이 작동 |
| Grid point | `xi == x[i]`일 때 모든 side 옵션에서 `y[i]` 반환 (`iszero(dt1)` 체크) |
| Boundary handling | `xi == x[end]` 특수 처리 (`:wrap` 제외 시 `y[end]` 직접 반환, `:wrap`은 `x_min`으로 wrap) |
| Extrap + side | `:constant`/`:extension` 외삽에서 side 무시, 경계값 직접 반환 |
| **Extrap + deriv** | **외삽 구간에서 `deriv=1/2`는 항상 `zero(T)` 반환 (`_constant_eval_extrap` op별 오버로드)** |
| Wrap domain | `[x_min, x_max)` 반개구간, `xi == x_max` → `x_min`으로 wrap |
| deriv keyword | `ConstantInterpolant` 호출에서 `deriv` 키워드 지원 (DerivativeView 호환) |
| Tolerance | `iszero(dt1)` strict 비교 (허용오차 없음) |
| **Real wrappers** | **Scalar, Vector, In-place, 2-arg callable 모두 Real → Float 변환 래퍼 제공** |
| **Real type policy** | **x,y는 동일 타입 `T<:Real` 요구, xi는 별도 `S<:Real` 허용 (linear_interp와 동일)** |
| **:none domain check** | **`@boundscheck _check_domain` 선행으로 DomainError 보장, `_constant_eval_extrap`에 Val{:none} 불필요** |
| **ConstantInterpolant API** | **Scalar, Real wrapper, Vector, In-place 모두 지원 (LinearInterpolant 패턴)** |
| **NaN/Inf policy** | **NaN/Inf 입력은 undefined behavior (caller's responsibility, 문서화만)** |
