# Tests that regression baselines are computed from SAME-MACHINE points only.
# Run: julia --project=benchmark benchmark/test/test_regression_machine_filter.jl

using Test, JSON, BenchmarkTools, Printf

include(joinpath(@__DIR__, "..", "bench_machine.jl"))
include(joinpath(@__DIR__, "..", "regression_check.jl"))

cpu(name, nc) = Dict{String, Any}(
    "cpu_name" => name, "ncores" => nc, "model" => "", "speed_mhz" => 0,
    "arch" => "x86_64", "cpu_threads" => nc, "julia" => "1.11"
)
bench(name, val) = Dict{String, Any}("name" => name, "value" => val, "unit" => "ns")
entry(sha, c, val, date) = Dict{String, Any}(
    "commit" => Dict{String, Any}("id" => sha), "date" => date,
    "benches" => [bench("a", val)], "cpu" => c
)

function write_fixture(path, suites)
    data = Dict{String, Any}("entries" => suites, "lastUpdate" => 0, "repoUrl" => "")
    return open(path, "w") do io
        print(io, "window.BENCHMARK_DATA = ")
        JSON.print(io, data)
        print(io, ";")
    end
end

const ZN = cpu("znver3", 4)
const SK = cpu("skylake-avx512", 4)

# Canonical suite = znver3 series; secondary suite = skylake series (much slower),
# with s1 shared across both machines. Interleaved dates.
function fixture_path()
    dir = mktempdir()
    p = joinpath(dir, "data.js")
    write_fixture(
        p, Dict{String, Any}(
            SUITE_NAME => Any[
                entry("s1", ZN, 100.0, 1),
                entry("s2", ZN, 110.0, 3),
                entry("s3", ZN, 120.0, 5),
            ],
            "$SUITE_NAME (skylake-avx512|4c)" => Any[
                entry("s1", SK, 500.0, 2),
                entry("s2", SK, 510.0, 4),
            ],
        )
    )
    return p
end

@testset "load_master_baseline: same-machine floor + baseline, skylake ignored" begin
    p = fixture_path()
    prev_best, latest, window_avg = load_master_baseline(p, "s3", "znver3|4c")
    @test prev_best["a"] == 120.0                 # same (sha=s3, machine=znver3) floor
    @test latest["a"] == 110.0                    # last OTHER znver3 commit, NOT skylake 510
    @test latest["a"] != 510.0
    @test all(v < 200.0 for v in values(window_avg))   # window never pulls in skylake's 500s
end

@testset "load_baseline: filters to the requested machine" begin
    p = fixture_path()
    lat_zn, _ = load_baseline(p, "znver3|4c")
    @test lat_zn["a"] == 120.0                    # last znver3 point (by date)
    lat_sk, _ = load_baseline(p, "skylake-avx512|4c")
    @test lat_sk["a"] == 510.0                    # last skylake point, NOT znver3's 120
end

@testset "first run on an unseen machine ⇒ empty baselines (graceful)" begin
    p = fixture_path()
    prev_best, latest, window_avg = load_master_baseline(p, "s9", "apple-m9|8c")
    @test isempty(prev_best)
    @test isempty(latest)
    @test isempty(window_avg)
end

@testset "latest_master_machine: machine of the most-recent master commit overall" begin
    # fixture_path's most recent point (by date) is s3 on znver3 (date 5).
    @test latest_master_machine(fixture_path()) == "znver3|4c"

    # A latest point without a cpu fingerprint ⇒ "unknown" (still lets us warn).
    dir = mktempdir()
    p = joinpath(dir, "data.js")
    no_cpu = Dict{String, Any}("commit" => Dict{String, Any}("id" => "s2"), "date" => 9, "benches" => [bench("a", 1.0)])
    write_fixture(p, Dict{String, Any}(SUITE_NAME => Any[entry("s1", ZN, 1.0, 1), no_cpu]))
    @test latest_master_machine(p) == "unknown"

    # No history ⇒ "" (⇒ no mismatch warning; the "first run" path applies).
    empty = joinpath(mktempdir(), "data.js")
    write_fixture(empty, Dict{String, Any}())
    @test latest_master_machine(empty) == ""
    @test latest_master_machine("/no/such/file.js") == ""
end

@testset "latest_master_machine: deterministic tie-break, prefers fingerprinted" begin
    # Same-commit points on two CPUs share the commit date (a max-date tie).
    # A fingerprinted CPU must win over an unfingerprinted "unknown" point.
    p = joinpath(mktempdir(), "data.js")
    no_cpu = Dict{String, Any}("commit" => Dict{String, Any}("id" => "s3"), "date" => 5, "benches" => [bench("a", 1.0)])
    write_fixture(
        p, Dict{String, Any}(
            SUITE_NAME => Any[entry("s1", ZN, 1.0, 1), entry("s3", ZN, 1.0, 5)],
            "$SUITE_NAME (unknown)" => Any[no_cpu],
        )
    )
    @test latest_master_machine(p) == "znver3|4c"

    # Two fingerprinted CPUs at the same max date resolve deterministically
    # (lexicographically smallest), not by suite/hash iteration order.
    p2 = joinpath(mktempdir(), "data.js")
    write_fixture(
        p2, Dict{String, Any}(
            SUITE_NAME => Any[entry("s3", ZN, 1.0, 5)],
            "$SUITE_NAME (skylake-avx512|4c)" => Any[entry("s3", SK, 1.0, 5)],
        )
    )
    @test latest_master_machine(p2) == "skylake-avx512|4c"   # "skylake…" < "znver3…"
end
