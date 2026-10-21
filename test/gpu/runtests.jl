# ========================================
# Real-GPU tests (opt-in): the backend named by FI_GPU_BACKEND
# ========================================
# Not part of CI or the default test run: CI has no GPU, and the GPU logic is covered there
# by the JLArrays tests in test/ext. These run real kernels on a real device: compilation,
# device numerics, kernel arguments. They are written against KernelAbstractions' backend
# interface (`adapt(backend, ·)`, `argconvert`, `supports_float64`), so one suite serves
# every GPUArrays backend:
#
#   FI_GPU_BACKEND=cuda julia test/gpu/runtests.jl
#   FI_GPU_BACKEND=cuda julia --project -e 'using Pkg; Pkg.test(test_args = ["gpu/runtests.jl"])'
#   FI_GPU_BACKEND=metal cc-julia-test-runner . gpu/runtests.jl
#
# FI_GPU_BACKEND: cuda (CUDA.jl), metal (Metal.jl), amdgpu (AMDGPU.jl), oneapi (oneAPI.jl).
# The suite runs in a fresh temporary environment: FastInterpolations from this checkout, the
# packages below and the backend. Nothing tracked changes, and the CPU test environment
# (Enzyme, Symbolics, …) is never resolved together with a GPU stack.

using Pkg

# Package, KernelAbstractions backend type, and the package's precision-changing converter
# (`cu`, `mtl`: Float64 → Float32; tests of it are skipped where there is none).
const GPU_BACKENDS = (
    cuda = (pkg = :CUDA, backend = :CUDABackend, narrow = :cu),
    metal = (pkg = :Metal, backend = :MetalBackend, narrow = :mtl),
    amdgpu = (pkg = :AMDGPU, backend = :ROCBackend, narrow = nothing),
    oneapi = (pkg = :oneAPI, backend = :oneAPIBackend, narrow = nothing),
)

const GPU_NAME = Symbol(lowercase(get(ENV, "FI_GPU_BACKEND", "")))
haskey(GPU_BACKENDS, GPU_NAME) ||
    error("set FI_GPU_BACKEND to one of $(join(keys(GPU_BACKENDS), ", ")) to run the GPU tests")
const GPU_SPEC = GPU_BACKENDS[GPU_NAME]

Pkg.activate(; temp = true)
Pkg.develop(path = normpath(joinpath(@__DIR__, "..", "..")))
Pkg.add([string(GPU_SPEC.pkg), "Adapt", "GPUArrays", "KernelAbstractions", "StaticArrays", "ForwardDiff", "Unitful", "Test", "InteractiveUtils"])

using Test
using FastInterpolations, Adapt, GPUArrays, KernelAbstractions
const FI = FastInterpolations
const KA = KernelAbstractions
const GPU_PKG = Base.require(Main, GPU_SPEC.pkg)
GPU_PKG.functional() || error("$(GPU_SPEC.pkg) is installed but finds no usable device")
const BACKEND = getfield(GPU_PKG, GPU_SPEC.backend)()

# Backend-agnostic helpers for the test files.
dev(a) = adapt(BACKEND, a)                                      # host array → device array
const GPUArrayType = Base.typename(typeof(dev(Float32[0]))).wrapper  # CuArray, MtlArray, …: `adapt(GPUArrayType, itp)`
@kernel function _argconvert_probe!(a) end
kernel_arg(x) = KA.argconvert(_argconvert_probe!(BACKEND), x)  # what a kernel receives
on_device(a) = get_backend(a) == BACKEND
const GPU_ELTYPES = KA.supports_float64(BACKEND) ? (Float32, Float64) : (Float32,)
const NARROW = GPU_SPEC.narrow === nothing ? nothing : getfield(GPU_PKG, GPU_SPEC.narrow)
const FLUSHES_SUBNORMALS = iszero(Array(dev([nextfloat(0.0f0)]) .* 1.0f0)[1])

@info "GPU tests" backend = GPU_NAME eltypes = GPU_ELTYPES flushes_subnormals = FLUSHES_SUBNORMALS converter = GPU_SPEC.narrow

include("linear1d.jl")
include("linear_nd.jl")
