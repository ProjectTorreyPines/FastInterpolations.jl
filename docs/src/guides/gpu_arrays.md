# [GPU Arrays](@id gpu_arrays)

Linear interpolants (1D and ND) run on GPUs through the GPUArrays interface, with no backend-specific code: the same code serves every GPUArrays backend (CUDA.jl, Metal.jl, AMDGPU.jl, oneAPI.jl; tested on CUDA and Metal). Loading a GPU package is enough: it loads Adapt.jl and GPUArrays.jl, which activate the GPU support.

The idea is simple: a GPU array is just another storage. Build an interpolant from arrays already on the device, pass device queries, and the usual API runs there — the same calls, keywords and results as on the CPU, with no data passing through the host.

## Supported

| | 1D `linear_interp(x, y)` | ND `linear_interp((x1, x2, …), data)` |
|---|---|---|
| Build from device arrays (scenario 1) | ✓ | ✓ |
| Move a CPU interpolant with `Adapt.adapt` (scenario 2) | ✓ | ✓ |
| Batch calls (allocating, in-place, any query shape) | ✓ | ✓ SoA (tuple of arrays) and AoS (array of points) |
| Broadcasts and calls inside your own kernels | ✓ | ✓ |
| Derivatives | `deriv = DerivOp(n)`, `deriv1`/`deriv2`/`deriv3` | per-axis `deriv`; `gradient`, `value_gradient`, `laplacian`, `jacobian` inside kernels |
| One-shot `linear_interp(x, y, xq)` | ✓ | — |
| Periodic boundary conditions | ✓ | — |

## Usage

### Scenario 1: the data is already on the device

The main case: the grid, the data and the queries live on the device (computed there, or loaded once). Pass device arrays where you would pass `Vector`s; results come back as device arrays.

```julia
using FastInterpolations, CUDA     # or Metal (MtlArray), AMDGPU (ROCArray), oneAPI (oneArray)

# x, y: vectors you already have on the device
x  = CuArray(range(0.0f0, 1.0f0, 1001) .^ 1.5f0)   # 1001-element CuArray{Float32}: the grid (here non-uniform)
y  = sin.(6 .* x)                                   # 1001-element CuArray{Float32}: its values
xq = CUDA.rand(Float32, 10^6)                       # 10^6 queries on the device

# One-shot evaluation
linear_interp(x, y, xq)            # → 1000000-element CuArray{Float32}

# In-place evaluation (no allocation)
out = similar(xq)
linear_interp!(out, x, y, xq)

# Create reusable interpolant
itp = linear_interp(x, y)
itp(xq)                            # → 1000000-element CuArray{Float32}
itp(out, xq)                       # in-place
itp.(xq)                           # broadcast
itp(xq; deriv = DerivOp(1))        # slopes
itp(reshape(xq, 1000, :))          # any query shape → 1000×1000 CuArray{Float32}
Array(out)                         # copy results to the host when needed

# A uniform grid can stay a range: a few scalars, no search per query
itp_r = linear_interp(range(0.0f0, 1.0f0, 1001), CUDA.rand(Float32, 1001))
```

ND grids are ranges or device vectors, the data a device array; queries are SoA (a tuple of coordinate arrays) or AoS (an array of points):

```julia
xg = range(0.0f0, 1.0f0, 513)                     # a range axis
yg = CuArray(range(0.0f0, 1.0f0, 257) .^ 1.5f0)   # a vector axis on the device
f  = linear_interp((xg, yg), CUDA.rand(Float32, 513, 257))   # data on the device

xs, ys = CUDA.rand(Float32, 10^6), CUDA.rand(Float32, 10^6)
f((xs, ys))                        # SoA → 1000000-element CuArray{Float32}
f(tuple.(xs, ys))                  # AoS: the same points as an array of tuples → the same values
f.(xs, ys)                         # broadcast
```

A uniform grid (any `AbstractRange`) is a few scalars and travels by value; a non-uniform grid and the data stay in device memory. Construction copies them on the device by default, so later changes to the caller's arrays do not reach the interpolant (`store = StorePolicy(copy = false)` references them instead, as on the CPU). No data passes through the host; a periodic boundary condition reads the two endpoint values once. Output and queries must both be on the device; a mix is an `ArgumentError`.

### Scenario 2: an interpolant built on the CPU

An interpolant built on the CPU moves to the device once with Adapt.jl, the standard way to move a structure between storages (Adapt comes with every GPU package; add it to your environment to call it). Afterwards the same calls as in scenario 1 apply:

```julia
using Adapt
x_cpu   = range(0.0f0, 1.0f0, 1001) .^ 1.5f0     # a Vector{Float32} on the host
itp_cpu = linear_interp(x_cpu, sin.(6 .* x_cpu))

ditp = adapt(CuArray, itp_cpu)          # to the device, same precision
ditp(xq)                                # → 1000000-element CuArray{Float32}
Array(ditp(xq)) ≈ itp_cpu(Array(xq))    # true: the CPU's values
adapt(Array, ditp)                      # back to the host

using KernelAbstractions: get_backend
ditp = adapt(get_backend(xq), itp_cpu)  # backend-agnostic: the backend of any device array
```

The backend packages also have converters that change the precision on the way: `cu` (CUDA) and `mtl` (Metal) turn `Float64` into `Float32`. A uniform grid converts as the converter would convert a vector grid, together with the data and the fill value.

### A single point

A host scalar, `itp(0.5f0)`, would read the device storage element by element, and GPUArrays disallows that by default (`Scalar indexing is disallowed`): in a host loop it would be orders of magnitude slower than the CPU. When one value is all you need:

| | one value, measured on an M1 Pro (Metal), n = 1001 |
|---|---|
| `only(Array(itp(CuArray([0.5f0]))))` | one kernel launch (≈ 160 µs on the M1 Pro) |
| `CUDA.@allowscalar itp(0.5f0)` | one transfer per element read: ≈ 0.2 ms (range grid) to 2.3 ms (vector grid, binary search) |
| `itp_host = adapt(Array, itp)`, then `itp_host(0.5f0)` | one copy of the storage to the host, then CPU speed (≈ 0.01 µs per value) |

`@allowscalar` comes with every backend package (`CUDA.@allowscalar`, `Metal.@allowscalar`, …); nothing else needs to be installed. For many points, pass them as one device array.

### Inside your own kernels

An interpolant on the device is also an ordinary kernel argument. It works in broadcasts, which fuse with the surrounding computation into one kernel (see [Extrapolation on the device](#Extrapolation-on-the-device) for `NoExtrap()`), and inside your own kernels:

```julia
using KernelAbstractions, StaticArrays

# E: a 3D CuArray of SVector{3,Float32} field values; pos, vel: CuArray{SVector{3,Float32}}
Ef = linear_interp((xg, yg, zg), E; extrap = FillExtrap(zero(SVector{3, Float32})))   # zero field outside the box

@kernel function advance!(pos, vel, Ef, dt)           # a simplified particle step
    i = @index(Global, Linear)
    @inbounds p = pos[i]
    @inbounds vel[i] += dt * Ef(p)                    # one SVector point → SVector field value
    @inbounds pos[i] = p + dt * vel[i]
end

advance!(get_backend(pos))(pos, vel, Ef, 1.0f-3; ndrange = length(pos))
```

## Extrapolation on the device

A kernel cannot throw an error with a message. Batch calls handle `NoExtrap()` (the default) for you: the evaluation kernel checks the domain as it goes, the host reads the result once afterwards, and an out-of-domain (or `NaN`) query throws the same `DomainError` as on the CPU. Unlike on the CPU, an in-place output may then already hold partial results. The check costs one wait for the device per call; the first such call says so once in a warning.

A broadcast of a `NoExtrap()` interpolant over device arrays, `itp.(xq)` (ND: `f.(pts)` or `f.(xs, ys)`; nested broadcasts such as `itp.(2 .* xq)` and scalar coordinates included), runs as that batch call: the same values and the same error, computed on its own rather than fused with the rest of the broadcast. Interpolants with any other extrapolation broadcast as usual, fused into one kernel.

Your own kernels evaluate one point at a time and need a non-throwing extrapolation: `ClampExtrap()`, `FillExtrap(v)`, `ExtendExtrap()`, `WrapExtrap()` or, for queries known to be in the domain, `InBounds()`. A `NoExtrap()` interpolant there fails to compile.

## Element types

Device arrays hold any isbits element, and the same evaluation code compiles for each:

- values: `Float32`, `Float64`, `Complex`, `SVector`, Unitful quantities;
- queries: real numbers, Unitful quantities, ForwardDiff duals;
- grids: real numbers or Unitful quantities.

Use `Float32` on GPUs without fast `Float64` (Metal has no `Float64` at all). The converters differ for static vectors: `mtl` converts `SVector{N, Float64}` elements to `Float32`, `cu` leaves them as they are; for `Float32` on CUDA, build the data as `SVector{N, Float32}`.

## Performance notes

- **Keep data on the device.** Grids and data move once; afterwards only queries and results cross the host–device link, at the PCIe bandwidth. When queries start and end on the CPU, those transfers usually dominate.
- **Batch size.** Each call launches at least one kernel (measured: about 10–20 µs on a datacenter CUDA GPU, 100–200 µs on an Apple M1 Pro): against one CPU thread, the GPU paid off from about 10⁴ queries on CUDA and 10⁵–10⁶ on the M1 Pro.
- **Uniform grids are cheapest**: the cell index is computed directly. A non-uniform grid adds a binary search per query.
- **Search policies.** Each query runs on its own thread, so no search hint carries from one query to the next: on a non-uniform grid the default `AutoSearch()` runs as `BinarySearch()`, and so do `LinearBinarySearch()`, `LinearGallopSearch()` and `LinearSearch()` (with a one-time warning). Sorted queries still run faster on large grids, through memory locality. Uniform grids take the direct lookup whatever the policy.
- **`NoExtrap()` costs one wait for the device per batch call** (its domain check): with queries known to lie in the domain, `InBounds()` skips it; `ClampExtrap()` and `FillExtrap(v)` never throw and never wait.

## Not supported yet

- Other interpolation families (cubic, quadratic, constant, Hermite) — a batch call on GPU queries throws an `ArgumentError`.
- `integrate`, `Series` interpolants, `GriddedQuery`, ND one-shot calls, ND periodic boundary conditions.
- `hessian` inside kernels (it returns a heap `Matrix`).
- `hint = Ref(…)` on batch calls over GPU queries. Inside your own kernels, a `Ref` created in the kernel works as a per-thread hint.
- Host queries on a device interpolant (`itp(xq_cpu)`); for a single point, see [A single point](#A-single-point).

## Testing on your GPU

CI has no GPU: it checks the GPU logic with JLArrays, which runs kernels on the CPU. The real-GPU tests (kernel compilation, device numerics, evaluation inside user kernels) are written against KernelAbstractions' backend interface and run on any supported device on request. Name the backend with `FI_GPU_BACKEND`: `cuda`, `metal`, `amdgpu` or `oneapi`.

```bash
FI_GPU_BACKEND=cuda julia --project -e 'using Pkg; Pkg.test(test_args = ["gpu/runtests.jl"])'
# or, from a checkout:
FI_GPU_BACKEND=cuda julia test/gpu/runtests.jl
```

The tests run in a fresh temporary environment that adds the backend package, so nothing needs to be installed first. `Float64` runs where the device supports it.
