# Shared machine-identity helper for the benchmark pipeline.
#
# GitHub-hosted runners are drawn at random from a mixed CPU fleet, so a run can
# land on a noticeably faster/slower box than a previous one. To compare only
# like-with-like we tag every measurement with a machine key and never merge or
# compare across keys. This file is intentionally **dependency-free** (only
# `Sys.*`) so it can be `include`d by the benchmark scripts and also invoked
# standalone from a fast CI shell step:
#
#     julia -e 'include("benchmark/bench_machine.jl"); print(machine_key())'

function hardware_fingerprint()
    ci = Sys.cpu_info()
    return Dict{String, Any}(
        "cpu_name" => Sys.CPU_NAME,          # LLVM uarch target (skylake, znver3, ...)
        "arch" => String(Sys.ARCH),
        "model" => isempty(ci) ? "" : ci[1].model,
        "speed_mhz" => isempty(ci) ? 0 : ci[1].speed,
        "ncores" => length(ci),
        "cpu_threads" => Sys.CPU_THREADS,
        "julia" => string(VERSION),
    )
end

# Canonical, JSON/filesystem-safe machine identity, e.g. "znver3|4c".
#
# Keyed on the LLVM microarchitecture (which determines the native code Julia
# emits, so same-key runs are genuinely comparable) plus the core count (runner
# tier). Deliberately excludes `speed_mhz` — on Linux it reflects the live
# turbo/thermal frequency and drifts within a single box, which would shatter
# one machine into phantom "machines". Also excludes the Julia version (Phase 1
# choice): a version bump is a legitimate step-change that should stay visible on
# the trend rather than fragment the floor.
machine_key(hw = hardware_fingerprint()) = string(hw["cpu_name"], "|", hw["ncores"], "c")
