# ========================================
# GPUArrays Extension for FastInterpolations.jl
# ========================================
# Recognizes GPU storage for every GPUArrays backend (CUDA, Metal, AMDGPU, oneAPI, JLArrays):
# host handles and wrapped ones such as strided views (`AnyGPUArray`). Interpolants built
# from or moved to such storage hold it in the device leaf `_DeviceArray`.

module FastInterpolationsGPUArraysExt

using FastInterpolations
using GPUArrays: AnyGPUArray

FastInterpolations._is_device_storage(::AnyGPUArray) = true

end # module
