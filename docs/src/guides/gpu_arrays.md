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

## Getting an interpolant on the device

### Scenario 1: the data is already on the device

The main case: the grid and the data live on the device (computed there, or loaded once), so the interpolant is built there. Pass device arrays where you would pass `Vector`s and `Array`s:

```julia
using FastInterpolations, CUDA        # or Metal (MtlArray), AMDGPU (ROCArray), oneAPI (oneArray)

x   = CuArray(Float32[(i / 1000)^1.5 for i in 0:1000])   # a non-uniform grid on the device
y   = sin.(6 .* x)                                        # data computed on the device
itp = linear_interp(x, y; extrap = ClampExtrap())

xr    = range(0.0f0, 1.0f0; length = 1001)                # a uniform grid stays a range
itp_r = linear_interp(xr, sin.(6 .* CuArray(collect(xr))); extrap = ClampExtrap())

# ND: grids are ranges or device vectors, the data a device array
yg = CuArray(Float32[(i / 256)^1.5 for i in 0:256])
f  = linear_interp((xr, yg), CUDA.rand(Float32, 1001, 257); extrap = ClampExtrap())
```

A uniform grid (any `AbstractRange`) is a few scalars and travels by value; a non-uniform grid and the data stay in device memory. Construction copies them on the device by default, so later changes to `x` or `y` do not reach the interpolant (`store = StorePolicy(copy = false)` references them instead, as on the CPU). No data passes through the host; a periodic boundary condition reads the two endpoint values once.

### Scenario 2: an interpolant built on the CPU

An interpolant built on the CPU moves to the device once with Adapt.jl, the standard way to move a structure between storages (Adapt comes with every GPU package; add it to your environment to call it):

```julia
using Adapt
itp_cpu = linear_interp(x_cpu, y_cpu; extrap = ClampExtrap())

ditp = adapt(CuArray, itp_cpu)          # to the device, same precision
adapt(Array, ditp)                      # back to the host

using KernelAbstractions: get_backend
ditp = adapt(get_backend(x), itp_cpu)   # backend-agnostic: the backend of any device array
```

The backend packages also have converters that change the precision on the way: `cu` (CUDA) and `mtl` (Metal) turn `Float64` into `Float32`. A uniform grid converts as the converter would convert a vector grid, together with the data and the fill value.

## Evaluating

Pass device queries to the usual calls:

```julia
xq  = CUDA.rand(Float32, 10^7)
out = itp(xq)                         # CuArray{Float32}
itp(out, xq)                          # in-place, no allocation
itp(xq; deriv = DerivOp(1))           # derivative
itp(reshape(xq, 1000, :))             # shaped queries keep their shape

linear_interp(x, y, xq; extrap = ClampExtrap())   # one-shot on device arrays: nothing kept, nothing copied

# ND: SoA (a tuple of coordinate arrays) or AoS (an array of points)
v   = f((CUDA.rand(Float32, 10^6), CUDA.rand(Float32, 10^6)))
pts = CuArray([(rand(Float32), rand(Float32)) for _ in 1:10^6])
v2  = f(pts)
```

Output and queries must both be on the device; a mix is an `ArgumentError`.

An interpolant on the device is also an ordinary kernel argument. It works in broadcasts, which fuse with the surrounding computation into one kernel, and inside your own kernels:

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

A kernel cannot throw an error with a message. Batch calls handle `NoExtrap()` (the default) for you: one device reduction checks the domain before the kernel runs, and an out-of-domain query throws the same `DomainError` as on the CPU, before any output is written. This check costs one extra pass over the queries and a host synchronization.

Broadcasts and your own kernels evaluate one point at a time and need a non-throwing extrapolation: `ClampExtrap()`, `FillExtrap(v)`, `ExtendExtrap()`, `WrapExtrap()` or, for queries known to be in the domain, `InBounds()`. A `NoExtrap()` interpolant there fails to compile.

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
- **`NoExtrap()` costs a domain check per batch call**: use `ClampExtrap()` or `InBounds()` when the queries are known to be valid.

## Not supported yet

- Other interpolation families (cubic, quadratic, constant, Hermite) — a batch call on GPU queries throws an `ArgumentError`.
- `integrate`, `Series` interpolants, `GriddedQuery`, ND one-shot calls, ND periodic boundary conditions.
- `hessian` inside kernels (it returns a heap `Matrix`), and search hints (`hint = Ref(…)`).
- Calling a device interpolant with host queries (`ditp(0.5)`, `ditp(xq_cpu)`).

## Testing on your GPU

CI has no GPU: it checks the GPU logic with JLArrays, which runs kernels on the CPU. The real-GPU tests (kernel compilation, device numerics, evaluation inside user kernels) are written against KernelAbstractions' backend interface and run on any supported device on request. Name the backend with `FI_GPU_BACKEND`: `cuda`, `metal`, `amdgpu` or `oneapi`.

```bash
FI_GPU_BACKEND=cuda julia --project -e 'using Pkg; Pkg.test(test_args = ["gpu/runtests.jl"])'
# or, from a checkout:
FI_GPU_BACKEND=cuda julia test/gpu/runtests.jl
```

The tests run in a fresh temporary environment that adds the backend package, so nothing needs to be installed first. `Float64` runs where the device supports it.
