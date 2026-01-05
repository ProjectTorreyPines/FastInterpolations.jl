# FastInterpolations.jl Benchmark CI Plan

> **Status:** ✅ Finalized
> **Last Updated:** 2026-01-04

## Overview

Implement automated benchmark tracking with historical visualization, inspired by [Metal.jl's benchmark system](https://metal.juliagpu.org/bench/).

### Goals
1. Track performance regressions automatically on PR/push
2. Visualize historical benchmark trends
3. Alert on significant performance degradation
4. Host results on GitHub Pages (no separate server)

### Design Decisions
- **Forward-only tracking:** No backfilling past commits (API compatibility issues)
- **PR feedback via comments:** No deploy previews (keeps Deployments section clean)
- **Focus on linear & cubic:** Most commonly used interpolation methods

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Push/PR/Manual │────▶│  GitHub Actions  │────▶│  Run Benchmarks │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  GitHub Pages   │◀────│  gh-pages branch │◀────│  output.json    │
│  /bench/        │     │  /bench/data.js  │     │  (BenchmarkTools)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│  Chart.js       │
│  Visualization  │
└─────────────────┘
```

---

## Components

### 1. Benchmark Script (`benchmark/ci_benchmark.jl`)

**Purpose:** Run standardized benchmarks and output JSON for CI consumption.

**Design Principles:**
- Focus on **linear** and **cubic** interpolation (most commonly used)
- Test with multiple query sizes: **n_query = [10, 100, 1000]**
- Fixed grid size: **n_grid = 100** (representative case)
- Both **one-shot** (construct + eval) and **evaluation** (reuse interpolant)

```julia
using BenchmarkTools
using FastInterpolations

suite = BenchmarkGroup()

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

const N_GRID = 100
const QUERY_SIZES = [10, 100, 1000]

# Setup data
x = range(0.0, 10.0, N_GRID)
y = sin.(x) .+ 0.1 .* collect(x)

# Pre-build interpolants for evaluation benchmarks
clear_cubic_cache!()
itp_linear = linear_interp(x, y)
itp_cubic = cubic_interp(x, y; autocache=false)

# ══════════════════════════════════════════════════════════════════════════════
# One-Shot Benchmarks (construct + evaluate)
# ══════════════════════════════════════════════════════════════════════════════
# Typical user workflow: interpolate once per dataset

for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

    suite["oneshot"]["linear_$nq"] = @benchmarkable linear_interp($x, $y, $xi)

    # Prime cache, then benchmark cache-hit performance
    clear_cubic_cache!()
    cubic_interp(x, y, xi)  # prime
    suite["oneshot"]["cubic_$nq"] = @benchmarkable cubic_interp($x, $y, $xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Benchmarks (reuse interpolant)
# ══════════════════════════════════════════════════════════════════════════════
# Performance when interpolant is reused across many evaluations

for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

    suite["eval"]["linear_$nq"] = @benchmarkable $itp_linear($xi)
    suite["eval"]["cubic_$nq"] = @benchmarkable $itp_cubic($xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Construction Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
# Track construction overhead separately

suite["construct"]["linear"] = @benchmarkable linear_interp($x, $y)

clear_cubic_cache!()
suite["construct"]["cubic"] = @benchmarkable cubic_interp($x, $y; autocache=false)

# ══════════════════════════════════════════════════════════════════════════════
# Package Comparison (cubic one-shot)
# ══════════════════════════════════════════════════════════════════════════════
# Compare against Interpolations.jl and DataInterpolations.jl
# Shows relative performance on same graph

import Interpolations
import DataInterpolations

const COMPARISON_QUERY_SIZES = [1, 10, 100, 1000]

for nq in COMPARISON_QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

    # FastInterpolations (cache-hit)
    clear_cubic_cache!()
    cubic_interp(x, y, xi)  # prime cache
    suite["compare"]["FastInterp_$nq"] = @benchmarkable cubic_interp($x, $y, $xi)

    # Interpolations.jl
    suite["compare"]["Interpolations_$nq"] = @benchmarkable begin
        itp = Interpolations.cubic_spline_interpolation($x, $y)
        itp($xi)
    end

    # DataInterpolations.jl
    suite["compare"]["DataInterp_$nq"] = @benchmarkable begin
        itp = DataInterpolations.CubicSpline($y, $x)
        itp($xi)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Run and Save
# ══════════════════════════════════════════════════════════════════════════════

tune!(suite)
results = run(suite, verbose=true)
BenchmarkTools.save("output.json", median(results))
```

**Benchmark Matrix:**

| Category | Benchmarks | Query Sizes | Total |
|----------|-----------|-------------|-------|
| One-Shot | linear, cubic | 10, 100, 1000 | 6 |
| Evaluation | linear, cubic | 10, 100, 1000 | 6 |
| Construction | linear, cubic | - | 2 |
| **Comparison** | FastInterp, Interpolations.jl, DataInterp | 1, 10, 100, 1000 | 12 |
| **Total** | | | **26** |

**Estimated Runtime:** ~2-3 minutes (0.5s per benchmark × 26 ≈ 13s + overhead)

**Visualization Note:** The comparison benchmarks will show 3 lines per query size on the dashboard, allowing direct performance comparison over time:
```
compare/FastInterp_1000      ──●── (fastest)
compare/Interpolations_1000  ──■──
compare/DataInterp_1000      ──▲──
```

---

### 2. GitHub Actions Workflow (`.github/workflows/Benchmark.yml`)

```yaml
name: Benchmark

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]
  workflow_dispatch:  # Manual trigger
    inputs:
      push_results:
        description: 'Push results to gh-pages'
        required: false
        default: 'false'
        type: boolean

permissions:
  contents: write
  pull-requests: write
  deployments: write

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Julia
        uses: julia-actions/setup-julia@v1
        with:
          version: '1'

      - name: Cache Julia artifacts
        uses: actions/cache@v4
        with:
          path: ~/.julia/artifacts
          key: ${{ runner.os }}-julia-${{ hashFiles('**/Project.toml') }}
          restore-keys: |
            ${{ runner.os }}-julia-

      - name: Install dependencies
        run: julia --project -e 'using Pkg; Pkg.instantiate()'

      - name: Run benchmarks
        run: julia --project benchmark/ci_benchmark.jl

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          name: FastInterpolations.jl Benchmarks
          tool: 'julia'
          output-file-path: output.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          # Push on: master push OR manual trigger with push_results=true
          auto-push: ${{ github.event_name == 'push' || (github.event_name == 'workflow_dispatch' && inputs.push_results == 'true') }}
          alert-threshold: '150%'
          comment-on-alert: true
          fail-on-alert: false
          gh-pages-branch: gh-pages
          benchmark-data-dir-path: bench
```

**Trigger Options:**

| Trigger | When | Pushes to gh-pages | PR Comment |
|---------|------|-------------------|------------|
| `push` | Merge to master | ✅ Yes | - |
| `pull_request` | PR opened/updated | ❌ No | ⚠️ On regression |
| `workflow_dispatch` | Manual via UI | Configurable | - |

**Manual Trigger Usage:**
1. Go to Actions tab → "Benchmark" workflow
2. Click "Run workflow"
3. Optionally check "Push results to gh-pages"

---

### 3. PR Feedback (Comment Only)

**Decision:** No deploy previews. Use PR comments only (keeps Deployments section clean).

**How it works:**
- `github-action-benchmark` compares PR results against baseline (latest master)
- If regression detected (>150%), posts a comment:

```
⚠️ Performance Alert

Benchmark       | Current | Previous | Ratio
----------------|---------|----------|------
cubic_1000      | 45.2 μs | 30.1 μs  | 1.50x ⚠️
linear_1000     | 12.3 μs | 12.1 μs  | 1.02x ✅
```

**Why no deploy preview:**
- Avoid Deployments section pollution
- Comment provides sufficient information for review
- Full chart history available on master's gh-pages after merge

---

### 4. GitHub Pages Configuration

The benchmark action creates its own `index.html` in the `bench/` directory. This coexists with Documenter.jl output.

```
gh-pages branch:
├── index.html          # Documenter.jl docs
├── interpolation/
├── api/
└── bench/
    ├── index.html      # Benchmark dashboard (auto-generated)
    └── data.js         # Historical data (auto-updated)
```

**Access URL:** `https://projecttorreypines.github.io/FastInterpolations.jl/bench/`

---

## Implementation Steps

### Phase 1: Basic Setup
1. [ ] Create `benchmark/ci_benchmark.jl` with core benchmarks
2. [ ] Create `.github/workflows/Benchmark.yml`
3. [ ] Test locally: `julia --project benchmark/ci_benchmark.jl`
4. [ ] Push to master and verify workflow runs
5. [ ] Test manual trigger via `workflow_dispatch`

### Phase 2: Verification
6. [ ] Confirm `gh-pages` branch has `/bench/` directory
7. [ ] Access benchmark dashboard at `https://projecttorreypines.github.io/FastInterpolations.jl/bench/`
8. [ ] Create test PR to verify regression comment works
9. [ ] Add link to benchmark page in documentation (optional)

### Phase 3: Tuning (After Data Collection)
10. [ ] Review alert threshold after ~10 data points
11. [ ] Consider adding derivative benchmarks if needed

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CI runner variance | False regression alerts | Use 150% threshold, median results |
| Workflow failures | No benchmark data | GitHub notifications on failure |
| gh-pages conflicts | Docs deployment issues | Separate `/bench/` path |
| Long benchmark runtime | Slow CI | 14 benchmarks @ 0.5s each ≈ 1-2 min total |
| data.js growth | Slow page load | Prune old data if >500 entries |

---

## Resolved Questions

| Question | Decision |
|----------|----------|
| PR Preview | ✅ Comment only (no deploy) |
| Alert threshold | ✅ Start with 150%, adjust later |
| Methods to track | ✅ linear + cubic (most used) |
| Query sizes | ✅ [10, 100, 1000] for core, [1, 10, 100, 1000] for comparison |
| Past commits | ✅ Forward-only (no backfill) |
| Comparison vs other libs | ✅ Include cubic one-shot comparison (3 packages × 4 sizes) |
| Derivatives | ⏸️ Defer (add later if needed) |

---

## References

- [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark)
- [Metal.jl Benchmark Workflow](https://github.com/JuliaGPU/Metal.jl/blob/main/.github/workflows/Benchmark.yml)
- [Metal.jl Benchmark Dashboard](https://metal.juliagpu.org/bench/)
- [BenchmarkTools.jl Documentation](https://juliaci.github.io/BenchmarkTools.jl/stable/)
