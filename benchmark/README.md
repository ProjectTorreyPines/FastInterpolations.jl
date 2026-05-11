# Benchmarks

Performance comparison against [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) and [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl).

## Setup

```bash
cd benchmark
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Configuration

The benchmark duration is set to `DEFAULT_BENCH_SECONDS = 0.5` in `simple_benchmarks.jl` for faster iteration. For more accurate results, increase this value to 5 or higher:

```julia
# In simple_benchmarks.jl, line 39
const DEFAULT_BENCH_SECONDS = 5.0  # More accurate results
```

Note: The result plots below were generated with `DEFAULT_BENCH_SECONDS = 5.0` or higher.

## Usage

```julia
using Pkg; Pkg.activate(".") # assuming `benchmark` is working directory
include("simple_benchmarks.jl")
include("plot_scaling.jl")

# Run full scaling benchmark
result = benchmark_scaling()

# Generate plots
plot_scaling_results(result) # combined plot
plot_scaling_separate(result; save_dir="../docs/images", dpi=250)
```

## Output

- `result.construction` - Construction time vs grid size (5 to 1000 points)
- `result.evaluation` - Evaluation time vs query points (1 to 10000, fixed grid n=100)
- `result.oneshot` - One-shot (construction + evaluation) time with cache-hit/miss comparison

## Results

### One-Shot (Construction + Evaluation)

![One-Shot](../docs/images/benchmark_oneshot_detail.png)

Combined construction + evaluation time per call (fixed grid size n=100, varying query points).

## CI Benchmarking (Regression Tests)

The `ci_benchmark.jl` script is used in the GitHub Actions CI workflow to monitor and prevent performance regressions. You can run it locally to test code changes against a baseline or profile specific components.

### Basic Usage

Run all benchmark groups in the suite:
```bash
cd benchmark && julia --project=. ci_benchmark.jl
```

### Filtering Groups

Since the full suite runs a large combination of benchmarks, you can easily control which groups are executed by passing arguments to the script. The script supports group numbers, exact group keys, or general substrings:

* **By Group Number** (runs only group 15, `15_phs_eval`):
  ```bash
  cd benchmark && julia --project=. ci_benchmark.jl 15
  ```

* **By Substring** (runs all 1D PHS-specific benchmark groups: `13_phs_oneshot`, `14_phs_construct`, and `15_phs_eval`):
  ```bash
  cd benchmark && julia --project=. ci_benchmark.jl phs
  ```

* **By Exact Key**:
  ```bash
  cd benchmark && julia --project=. ci_benchmark.jl 15_phs_eval
  ```

* **Combining Multiple Filters** (runs group 9 and group 15):
  ```bash
  cd benchmark && julia --project=. ci_benchmark.jl 9 15
  ```

### Comparing Against Baselines

To verify performance against a previously saved JSON baseline and trigger automated regression checks:
```bash
cd benchmark && julia --project=. ci_benchmark.jl --baseline output.json
```