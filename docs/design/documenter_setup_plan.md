# Documenter.jl Setup Plan for FastInterpolations.jl

## Overview

Documenter.jl을 사용하여 FastInterpolations.jl의 공식 문서 페이지를 생성하고 GitHub Pages에 배포하는 계획.

**핵심 원칙**: Single Source of Truth (DRY) - 콘텐츠 중복 없이 자동화된 빌드

---

## 기존 문서 분석

### 현재 파일 구조

```
FastInterpolations.jl/
├── README.md                              # 사용자용 메인 문서
├── benchmark/
│   └── README.md                          # 벤치마크 가이드
└── docs/
    ├── design/
    │   ├── derivative_design.md           # 내부 설계: 수학적 기초, 아키텍처
    │   └── interpolant_derivative_api.md  # 내부 설계: Hybrid B+C API
    └── images/                            # 벤치마크 그래프들
```

### 발견된 이슈

1. **Outdated**: `interpolant_derivative_api.md`에 `derivative(itp)` → `deriv1(itp)`로 업데이트 필요
2. **분리 필요**: Design docs(개발자) vs Documenter pages(사용자)

---

## 통합 전략 (Revised)

### DRY 원칙 적용

| 문제점 | 기존 접근 | 개선된 접근 |
|--------|----------|------------|
| README ↔ index.md 중복 | 수동으로 두 파일 관리 | `make.jl`에서 자동 복사 |
| 튜토리얼 코드 부패 | Markdown에 코드 수동 작성 | `Literate.jl`로 실행 가능한 `.jl`에서 생성 |
| 벤치마크 문서 중복 | 별도 파일 작성 | `benchmark/README.md` 자동 복사 |

### 자동화 콘텐츠 흐름

```
┌──────────────────────────────────────────────────────────────────────┐
│  Source Files (Single Source of Truth)                               │
├──────────────────────────────────────────────────────────────────────┤
│  README.md ──[경로 재작성]──────────> docs/src/index.md              │
│  benchmark/README.md ──[경로 재작성]> docs/src/guides/performance.md │
│  examples/*.jl ──[Literate.jl]──────> docs/src/tutorials/*.md        │
│  src/*.jl docstrings ──[@docs]──────> docs/src/api/*.md              │
│  docs/design/*.md ──[링크 페이지]───> docs/src/internals.md          │
└──────────────────────────────────────────────────────────────────────┘
```

**주의**: README 파일의 상대 경로(`docs/images/`, `../docs/images/`)는 Documenter 구조와 맞지 않으므로 복사 시 경로 재작성 필요.

### 최종 구조

```
docs/
├── Project.toml          # Documenter + Literate 의존성
├── make.jl               # 자동화된 빌드 스크립트
├── src/
│   ├── index.md          # (자동 생성) README.md 복사 + 경로 재작성
│   ├── tutorials/        # (자동 생성) Literate.jl로 examples/*.jl에서 생성
│   ├── guides/
│   │   └── performance.md  # (자동 생성) benchmark/README.md 복사 + 경로 재작성
│   ├── internals.md      # (수동 관리) design docs 링크 페이지
│   └── api/              # (수동 관리) @docs 매크로 사용
│       ├── linear.md
│       ├── cubic.md
│       └── types.md
├── design/               # 개발자용 (GitHub에서 직접 열람, 링크로 노출)
└── images/               # 공유 이미지
```

---

## Phase 1: 기본 구조 설정

### 1.1 `docs/Project.toml` 생성

```toml
[deps]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
FastInterpolations = "9ea80cae-fc13-4c00-8066-6eaedb12f34b"
Literate = "98b081ad-f1c9-55d3-8b20-4c87d4299306"

[compat]
Documenter = "1"
Literate = "2"
```

### 1.2 `docs/make.jl` 생성 (자동화된 버전)

```julia
using Documenter
using FastInterpolations
using Literate

# ============================================
# Helper: Rewrite relative paths in README files
# ============================================

"""
README.md의 상대 경로를 Documenter 구조에 맞게 재작성.
- `docs/images/` → `../images/` (index.md 기준)
- `../docs/images/` → `../images/` (guides/performance.md 기준)
- `benchmark/` → GitHub 절대 URL
"""
function rewrite_readme_paths(content::String; from_root::Bool=true)
    repo_url = "https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master"

    if from_root
        # README.md → index.md (docs/src/ 기준)
        content = replace(content, "docs/images/" => "images/")
        content = replace(content, "benchmark/" => "$(repo_url)/benchmark/")
    else
        # benchmark/README.md → guides/performance.md (docs/src/guides/ 기준)
        content = replace(content, "../docs/images/" => "../images/")
    end
    return content
end

# ============================================
# Step 1: Auto-generate content (DRY principle)
# ============================================

const DOCS_SRC = joinpath(@__DIR__, "src")
const EXAMPLES_DIR = joinpath(@__DIR__, "../examples")
const TUTORIALS_DIR = joinpath(DOCS_SRC, "tutorials")

# 디렉토리 생성 (초기 상태에서 없을 수 있음)
mkpath(DOCS_SRC)
mkpath(joinpath(DOCS_SRC, "guides"))
mkpath(TUTORIALS_DIR)

# 1a. Copy README.md → index.md (경로 재작성 적용)
readme_content = read(joinpath(@__DIR__, "../README.md"), String)
write(joinpath(DOCS_SRC, "index.md"), rewrite_readme_paths(readme_content; from_root=true))

# 1b. Copy benchmark/README.md → guides/performance.md (경로 재작성 적용)
bench_readme = joinpath(@__DIR__, "../benchmark/README.md")
if isfile(bench_readme)
    bench_content = read(bench_readme, String)
    write(joinpath(DOCS_SRC, "guides/performance.md"),
          rewrite_readme_paths(bench_content; from_root=false))
end

# 1c. Generate tutorials from examples/*.jl using Literate.jl
if isdir(EXAMPLES_DIR)
    for file in sort(readdir(EXAMPLES_DIR))  # 정렬하여 결정적 순서 보장
        if endswith(file, ".jl")
            Literate.markdown(
                joinpath(EXAMPLES_DIR, file),
                TUTORIALS_DIR;
                documenter = true,
                credit = false
            )
        end
    end
end

# ============================================
# Step 2: Build documentation
# ============================================

# Collect generated tutorial pages (정렬된 순서)
tutorial_pages = if isdir(EXAMPLES_DIR) && !isempty(readdir(EXAMPLES_DIR))
    sort(["tutorials/$(replace(f, ".jl" => ".md"))"
          for f in readdir(EXAMPLES_DIR) if endswith(f, ".jl")])
else
    String[]
end

makedocs(
    sitename = "FastInterpolations.jl",
    authors = "Min-Gu Yoo",
    modules = [FastInterpolations],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://projecttorreypines.github.io/FastInterpolations.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => tutorial_pages,
        "Guides" => [
            "Performance" => "guides/performance.md",
        ],
        "API Reference" => [
            "Linear" => "api/linear.md",
            "Cubic" => "api/cubic.md",
            "Types" => "api/types.md",
        ],
        "Internals" => "internals.md",
    ],
    doctest = true,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/ProjectTorreyPines/FastInterpolations.jl.git",
    devbranch = "master",
    push_preview = true,
)
```

### 1.3 `docs/src/internals.md` (Design Docs 링크 페이지)

```markdown
# Internals & Design Documents

이 섹션은 개발자를 위한 내부 설계 문서 링크를 제공합니다.

## Design Documents

아래 문서들은 GitHub 저장소에서 직접 열람할 수 있습니다:

- [Derivative Design](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/docs/design/derivative_design.md): 미분 계산의 수학적 기초와 커널 구현
- [Interpolant Derivative API](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/docs/design/interpolant_derivative_api.md): Hybrid B+C API 설계 결정

## Architecture Overview

FastInterpolations.jl의 내부 아키텍처:

- **Operation Types** (`src/ops.jl`): `EvalValue`, `EvalDeriv1`, `EvalDeriv2` 트레이트
- **Kernel Functions** (`src/*_kernels.jl`): 순수 수학 함수
- **Dispatch Macros** (`src/utils.jl`): 런타임→컴파일타임 변환
```

### 1.4 Examples 디렉토리 구조 (Literate.jl용)

```
examples/
├── 01_basic_usage.jl         # 기본 사용법 튜토리얼
├── 02_derivatives.jl         # 미분 계산 튜토리얼
└── 03_boundary_conditions.jl # BC 타입 튜토리얼
```

**Literate.jl 형식 예시** (`examples/01_basic_usage.jl`):

```julia
# # Basic Usage
#
# This tutorial covers the basic usage of FastInterpolations.jl.
#
# ## Linear Interpolation

using FastInterpolations

x = range(0.0, 10.0, 100)
y = sin.(x)

# Interpolate at a single point:
result = linear_interp(x, y, 5.5)

# Interpolate at multiple points:
queries = [1.0, 2.0, 3.0]
results = linear_interp(x, y, queries)

# ## Cubic Spline Interpolation
#
# Cubic splines provide smoother interpolation:

result_cubic = cubic_interp(x, y, 5.5)
```

---

## Phase 2: API Reference 페이지 (수동 관리)

### 2.1 `docs/src/api/linear.md`

```markdown
# Linear Interpolation API

## Functions

```@docs
linear_interp
linear_interp!
```

## Interpolant Type

```@docs
LinearInterpolant
```
```

### 2.2 `docs/src/api/cubic.md`

```markdown
# Cubic Spline API

## Functions

```@docs
cubic_interp
cubic_interp!
```

## Interpolant Type

```@docs
CubicInterpolant
CubicSplineCache
```

## Cache Management

```@docs
set_cubic_cache_size!
get_cubic_cache_size
clear_cubic_cache!
cubic_cache_stats
```

## Derivative Views

```@docs
deriv1
deriv2
```
```

### 2.3 `docs/src/api/types.md`

```markdown
# Type Reference

## Boundary Conditions

```@docs
AbstractBC
PointBC
Deriv1
Deriv2
BCPair
NaturalBC
ClampedBC
PeriodicBC
```

## Index

```@index
```
```

---

## Phase 3: GitHub Actions CI 설정

### 3.1 `.github/workflows/Documentation.yml`

```yaml
name: Documentation

on:
  push:
    branches:
      - master
    tags:
      - 'v*'
  pull_request:
    branches:
      - master

jobs:
  build:
    permissions:
      contents: write
      statuses: write
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1'
      - name: Install dependencies
        run: julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
      - name: Build and deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
        run: julia --project=docs/ docs/make.jl
```

### 3.2 Deploy Key 설정

```julia
using DocumenterTools
DocumenterTools.genkeys(user="ProjectTorreyPines", repo="FastInterpolations.jl")
```

---

## Phase 4: `.gitignore` 업데이트

```gitignore
# Documentation build
docs/build/
docs/Manifest.toml

# Auto-generated docs (created by make.jl)
docs/src/index.md
docs/src/tutorials/
docs/src/guides/performance.md
```

**중요**: 자동 생성 파일은 `.gitignore`에 추가하여 중복 관리 방지

---

## Implementation Checklist

### Phase 1: 기본 구조
- [ ] `docs/Project.toml` 생성 (Documenter + Literate)
- [ ] `docs/make.jl` 생성 (자동화 버전)
- [ ] `examples/` 디렉토리 및 Literate 형식 튜토리얼 작성
- [ ] `.gitignore` 업데이트

### Phase 2: 수동 관리 페이지
- [ ] `docs/src/api/linear.md` 작성
- [ ] `docs/src/api/cubic.md` 작성
- [ ] `docs/src/api/types.md` 작성
- [ ] `docs/src/internals.md` 작성 (Design docs 링크)
- [ ] 모든 export 심볼에 docstring 확인

### Phase 3: CI/CD
- [ ] `.github/workflows/Documentation.yml` 생성
- [ ] Deploy key 설정 (`DOCUMENTER_KEY`)
- [ ] GitHub Pages 활성화

### Phase 4: 검증
- [ ] 로컬 빌드 성공: `julia --project=docs docs/make.jl`
- [ ] Doctest 통과
- [ ] CI 빌드 및 배포 성공

---

## Notes

### Docstring 요구사항

`checkdocs = :exports` 사용 시, 모든 export 심볼에 docstring 필요:

```julia
# 현재 exports:
export linear_interp, linear_interp!, LinearInterpolant
export cubic_interp, cubic_interp!, CubicSplineCache, CubicInterpolant
export set_cubic_cache_size!, get_cubic_cache_size, clear_cubic_cache!, cubic_cache_stats
export AbstractBC, PointBC, Deriv1, Deriv2, BCPair
export NaturalBC, ClampedBC, PeriodicBC
export deriv1, deriv2
```

### Docstring에 예제 포함 권장

```julia
"""
    cubic_interp(x, y, xi; deriv=0, extrap=:none, bc=NaturalBC())

Evaluate cubic spline at query point(s).

# Examples
```jldoctest
julia> x = [0.0, 1.0, 2.0];
julia> y = [0.0, 1.0, 0.0];
julia> cubic_interp(x, y, 0.5)
0.625
```
"""
function cubic_interp(...)
```

Documenter가 `jldoctest` 블록을 자동 테스트하여 코드 부패 방지.

---

## 개선된 접근 요약

| 항목 | Before (Original) | After (Improved) |
|------|-------------------|------------------|
| **Home Page** | 수동으로 `index.md` 작성 | `README.md` 자동 복사 |
| **Tutorials** | 수동으로 Markdown 작성 | `Literate.jl`로 `.jl`에서 생성 |
| **Performance Guide** | 별도 작성 | `benchmark/README.md` 자동 복사 |
| **API Docs** | `@docs` 블록 사용 | `@docs` 블록 사용 (유지) |
| **유지보수** | 여러 파일 수동 동기화 | Single Source of Truth |

---

## Priority Order

1. **MVP**: Phase 1 + API reference (`@docs` 블록)
2. **확장**: `examples/*.jl` 튜토리얼 추가
3. **고급**: `jldoctest` 예제 추가, Cross-references

---

## Review Fixes Applied

| Severity | Issue | Fix |
|----------|-------|-----|
| **High** | README 상대 경로 깨짐 | `rewrite_readme_paths()` 함수로 경로 재작성 |
| **Medium** | `tutorial_pages` 가드가 잘못된 디렉토리 체크 | `isdir(EXAMPLES_DIR)` 조건으로 수정 |
| **Medium** | `docs/src/` 초기 상태에서 없을 수 있음 | `mkpath(DOCS_SRC)` 추가 |
| **Medium** | Design docs가 완전히 제외됨 | `internals.md` 링크 페이지 추가 |
| **Low** | `tutorial_pages` 순서 비결정적 | `sort()` 적용 |
| **Low** | 다이어그램에서 `perf.md` vs `performance.md` 불일치 | `performance.md`로 통일 |
