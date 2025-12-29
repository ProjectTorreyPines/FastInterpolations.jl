#=
Minimal Thread-Safety Test for cubic_autocache.jl

Run with: julia -t 4 --project test/minimal_thread_test.jl

예상되는 문제:
- 여러 스레드가 같은 cache의 workspace에 동시에 쓰면 데이터 손상 발생
- autocache=false로 하면 각자 cache를 만들어서 문제 없음
=#

using FastInterpolations
using Base.Threads

println("Julia threads: $(nthreads())")
nthreads() == 1 && println("⚠️  1개 스레드! `julia -t 4`로 실행하세요")

# 공유 grid - 모든 스레드가 같은 cache를 받음
x = collect(range(0.0, 1.0, 51))

# 결과 저장
N = 500
results_cached = Vector{Float64}(undef, N)
results_nocache = Vector{Float64}(undef, N)

# autocache 초기화 & cache 미리 생성 (race condition in push! 방지)
FastInterpolations.clear_cubic_cache!()
# _ = cubic_interp(x, sin.(x), 0.5; autocache=true)  # prime the cache

println("Cache primed, starting multi-threaded test...")

# 테스트: 각 iteration마다 다른 amplitude의 y 사용
@threads for i in 1:N
    amplitude = Float64(mod1(i, 10))  # 1.0 ~ 10.0
    y = amplitude .* sin.(2π .* x)

    # 두 가지 방식으로 같은 점 보간
    val_cached = cubic_interp(x, y, 0.25; autocache=true)
    val_nocache = cubic_interp(x, y, 0.25; autocache=false)

    results_cached[i] = val_cached
    results_nocache[i] = val_nocache
end

# 비교: cached vs nocache
errors = abs.(results_cached .- results_nocache)
n_errors = count(e -> e > 1e-10, errors)

println("\n결과:")
println("  총 테스트: $N")
println("  불일치 개수: $n_errors")
println("  최대 오차: $(maximum(errors))")

if n_errors > 0
    println("\n❌ 데이터 손상 발생!")
    println("   → workspace가 다른 스레드에 의해 덮어씌워짐")

    # 손상된 예시 몇 개 출력
    bad_idx = findall(e -> e > 1e-10, errors)
    println("\n손상된 예시 (처음 5개):")
    for i in bad_idx[1:min(5, length(bad_idx))]
        amp = Float64(mod1(i, 10))
        println("  i=$i (amp=$amp): cached=$(results_cached[i]), expected=$(results_nocache[i]), diff=$(errors[i])")
    end
else
    println("\n✅ 문제 없음 (더 많은 스레드/반복 필요할 수 있음)")
end
