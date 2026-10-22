# [GPU Arrays](@id gpu_arrays)

!!! warning "Experimental"
    GPU support may change in any release without deprecation: the supported families, backends and element types, `NoExtrap()` on the device, broadcast and search-policy handling, and the `adapt` and precision rules. CPU behaviour is unaffected.

Linear interpolants (1D and ND) run on any GPUArrays backend (CUDA.jl, Metal.jl, AMDGPU.jl, oneAPI.jl; tested on CUDA and Metal); loading a GPU package activates the support. A GPU array is just another storage: build from device arrays, pass device queries, and the usual API runs on the device with the same results as on the CPU.

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

Pass device arrays where you would pass `Vector`s; results come back as device arrays.

```julia
using FastInterpolations, CUDA     # or Metal (MtlArray), AMDGPU (ROCArray), oneAPI (oneArray)

x  = CuArray(range(0.0f0, 1.0f0, 1001) .^ 1.5f0)   # the grid (here non-uniform), on the device
y  = sin.(6 .* x)                                   # its values
xq = CUDA.rand(Float32, 10^6)                       # 10^6 queries

linear_interp(x, y, xq)            # one-shot → 1000000-element CuArray{Float32}
out = similar(xq)
linear_interp!(out, x, y, xq)      # in-place

itp = linear_interp(x, y)          # reusable interpolant
itp(xq)                            # → 1000000-element CuArray{Float32}
itp(out, xq)                       # in-place
itp.(xq)                           # broadcast
itp(xq; deriv = DerivOp(1))        # slopes
itp(reshape(xq, 1000, :))          # any query shape → 1000×1000 CuArray{Float32}
Array(out)                         # copy results to the host when needed

# A uniform grid can stay a range: a few scalars, no search per query
itp_r = linear_interp(range(0.0f0, 1.0f0, 1001), CUDA.rand(Float32, 1001))
```

ND grids are ranges or device vectors; queries are SoA (a tuple of coordinate arrays) or AoS (an array of points):

```julia
xg = range(0.0f0, 1.0f0, 513)                     # a range axis
yg = CuArray(range(0.0f0, 1.0f0, 257) .^ 1.5f0)   # a vector axis on the device
f  = linear_interp((xg, yg), CUDA.rand(Float32, 513, 257))

xs, ys = CUDA.rand(Float32, 10^6), CUDA.rand(Float32, 10^6)
f((xs, ys))                        # SoA → 1000000-element CuArray{Float32}
f(tuple.(xs, ys))                  # AoS: the same points → the same values
f.(xs, ys)                         # broadcast
```

Construction copies non-uniform grids and data on the device (`store = StorePolicy(copy = false)` references them, as on the CPU); nothing passes through the host. Output and queries must both be on the device.

### Scenario 2: an interpolant built on the CPU

Move it once with Adapt.jl (it comes with every GPU package; add it to your environment to call it), then use the same calls:

```julia
using Adapt
x_cpu   = range(0.0f0, 1.0f0, 1001) .^ 1.5f0     # a Vector{Float32} on the host
itp_cpu = linear_interp(x_cpu, sin.(6 .* x_cpu))

ditp = adapt(CuArray, itp_cpu)          # to the device, same precision
ditp(xq)                                # → 1000000-element CuArray{Float32}
adapt(Array, ditp)                      # back to the host

using KernelAbstractions: get_backend
ditp = adapt(get_backend(xq), itp_cpu)  # backend-agnostic
```

`cu` (CUDA) and `mtl` (Metal) work too and convert `Float64` to `Float32` on the way.

### A single point

A host scalar call, `itp(0.5f0)`, reads device memory element by element, which GPUArrays disallows by default. For one value:

| | cost (M1 Pro, n = 1001) |
|---|---|
| `only(Array(itp(CuArray([0.5f0]))))` | one kernel launch, ≈ 160 µs |
| `CUDA.@allowscalar itp(0.5f0)` | one transfer per element read: 0.2–2.3 ms |
| `adapt(Array, itp)(0.5f0)` | one copy of the storage to the host, then CPU speed |

For many points, pass them as one device array.

### Inside your own kernels

A device interpolant is an ordinary kernel argument, in broadcasts and in your own kernels:

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

A kernel cannot throw an error with a message, so:

- **Batch calls** handle `NoExtrap()` (the default): the kernel checks the domain and the host then throws the CPU's `DomainError` (an in-place output may already hold partial results). It costs one wait for the device per call, which a one-time warning points out.
- **Broadcasts** of a `NoExtrap()` interpolant over device arrays (nested broadcasts included) run as that batch call; other extrapolations broadcast as usual, fused into one kernel.
- **Your own kernels** need an extrapolation that cannot throw: `ClampExtrap()`, `FillExtrap(v)`, `ExtendExtrap()`, `WrapExtrap()`, or `InBounds()` for queries known to be in the domain. `NoExtrap()` does not compile there.

## Element types

Any isbits element works: values `Float32`, `Float64`, `Complex`, `SVector` or Unitful quantities; queries real numbers, Unitful quantities or ForwardDiff duals; grids real numbers or Unitful quantities. Use `Float32` on GPUs without fast `Float64` (Metal has none). `mtl` converts `SVector{N, Float64}` elements to `Float32` while `cu` keeps them, so build `SVector{N, Float32}` data for `Float32` on CUDA.

## Performance notes

- **Keep data on the device**: when queries start and end on the CPU, the transfers usually dominate.
- **Batch size**: each call is at least one kernel launch (about 10–20 µs on CUDA, 100–200 µs on an M1 Pro); against one CPU thread the GPU pays off from about 10⁴ queries on CUDA and 10⁵–10⁶ on the M1 Pro.
- **Uniform grids are cheapest** (the cell index is computed directly); a non-uniform grid adds a binary search per query.
- **Search policies**: each query runs on its own thread, so no search hint carries between queries; hint-based policies (`LinearBinarySearch()`, `LinearGallopSearch()`, `LinearSearch()`) run as `BinarySearch()`, with a one-time warning.
- **`NoExtrap()` waits for the device once per batch call**; `InBounds()` (queries known to be in the domain), `ClampExtrap()` and `FillExtrap(v)` do not.

## Not supported yet

- Other interpolation families (cubic, quadratic, constant, Hermite): a batch call on GPU queries throws an `ArgumentError`.
- `integrate`, `Series` interpolants, `GriddedQuery`, ND one-shot calls, ND periodic boundary conditions.
- `hessian` inside kernels (it returns a heap `Matrix`).
- `hint = Ref(…)` on batch calls; inside your own kernels, a `Ref` created in the kernel works as a per-thread hint.
- Host queries on a device interpolant (`itp(xq_cpu)`); see [A single point](#A-single-point).
- An in-place output sharing memory with the grid or data (see [Output aliasing](@ref memory_allocation)): unchecked, as on the CPU, and a data race on the GPU.

## Testing on your GPU

CI checks the GPU logic with JLArrays, which runs kernels on the CPU. The real-GPU tests run on request, in a fresh temporary environment that adds the backend package; name it with `FI_GPU_BACKEND` (`cuda`, `metal`, `amdgpu` or `oneapi`):

```bash
FI_GPU_BACKEND=cuda julia --project -e 'using Pkg; Pkg.test(test_args = ["gpu/runtests.jl"])'
# or, from a checkout:
FI_GPU_BACKEND=cuda julia test/gpu/runtests.jl
```
