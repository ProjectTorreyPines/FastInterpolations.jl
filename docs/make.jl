using Documenter
using FastInterpolations

# ============================================
# Helper: Conditional write (for LiveServer compatibility)
# ============================================

"""
Write file only if content changed (prevents LiveServer infinite loop).
"""
function write_if_changed(path::String, content::String)
    if isfile(path) && read(path, String) == content
        return  # 내용이 같으면 아무것도 하지 않음
    end
    write(path, content)
end

"""
Copy file only if content changed (prevents mtime update triggering rebuild).
"""
function cp_if_changed(src::String, dst::String)
    if isfile(dst) && read(src) == read(dst)
        return  # 내용이 같으면 복사 건너뜀
    end
    cp(src, dst; force=true)
end

# ============================================
# Helper: Rewrite relative paths in README files
# ============================================

"""
Rewrite relative paths in README.md for Documenter structure.
- `docs/images/` → `images/` (for index.md)
- `../docs/images/` → `../images/` (for guides/performance.md)
- `benchmark/` → GitHub absolute URL
"""
function rewrite_readme_paths(content::String; from_root::Bool=true)
    repo_url = "https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master"

    if from_root
        # README.md → index.md (docs/src/ base)
        content = replace(content, "docs/images/" => "images/")
        content = replace(content, "benchmark/" => "$(repo_url)/benchmark/")
    else
        # benchmark/README.md → guides/performance.md (docs/src/guides/ base)
        content = replace(content, "../docs/images/" => "../images/")
    end
    return content
end

# ============================================
# Step 1: Setup directories and content
# ============================================

const DOCS_SRC = joinpath(@__DIR__, "src")

# Create directories
mkpath(DOCS_SRC)
mkpath(joinpath(DOCS_SRC, "guides"))
mkpath(joinpath(DOCS_SRC, "interpolation"))
mkpath(joinpath(DOCS_SRC, "architecture"))

# Copy images directory
const IMAGES_SRC = joinpath(@__DIR__, "images")
const IMAGES_DST = joinpath(DOCS_SRC, "images")
if isdir(IMAGES_SRC)
    mkpath(IMAGES_DST)
    for img in readdir(IMAGES_SRC)
        src = joinpath(IMAGES_SRC, img)
        dst = joinpath(IMAGES_DST, img)
        if isfile(src)
            cp_if_changed(src, dst)
        end
    end
end

# Copy README.md → index.md (with path rewriting, conditional write)
readme_content = read(joinpath(@__DIR__, "../README.md"), String)
write_if_changed(joinpath(DOCS_SRC, "index.md"), rewrite_readme_paths(readme_content; from_root=true))

# Copy benchmark/README.md → guides/performance.md (with path rewriting, conditional write)
bench_readme = joinpath(@__DIR__, "../benchmark/README.md")
if isfile(bench_readme)
    bench_content = read(bench_readme, String)
    write_if_changed(joinpath(DOCS_SRC, "guides/performance.md"),
                     rewrite_readme_paths(bench_content; from_root=false))
end

# ============================================
# Step 2: Build documentation
# ============================================

makedocs(
    sitename = "FastInterpolations.jl",
    authors = "Min-Gu Yoo",
    modules = [FastInterpolations],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://projecttorreypines.github.io/FastInterpolations.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "API Selection" => "guides/api_selection.md",
            "Performance Tips" => "guides/performance_tips.md",
        ],
        "Interpolation" => [
            "Overview" => "interpolation/overview.md",
            "Constant" => "interpolation/constant.md",
            "Linear" => "interpolation/linear.md",
            "Quadratic" => "interpolation/quadratic.md",
            "Cubic" => "interpolation/cubic.md",
            "Derivatives" => "interpolation/derivatives.md",
            "Visual Comparison" => "interpolation/comparison.md",
        ],
        "Extrapolation" => "extrapolation.md",
        "Architecture" => [
            "Overview" => "architecture/overview.md",
            "Auto-Cache" => "architecture/caching.md",
            "Thread Safety" => "architecture/thread_safety.md",
        ],
        "API Reference" => [
            "Constant" => "api/constant.md",
            "Linear" => "api/linear.md",
            "Quadratic" => "api/quadratic.md",
            "Cubic" => "api/cubic.md",
            "Types" => "api/types.md",
        ],
        "Internals" => [
            "Overview" => "internals.md",
            "Benchmarks" => "guides/performance.md",
        ],
    ],
    doctest = true,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/ProjectTorreyPines/FastInterpolations.jl.git",
    devbranch = "master",
    push_preview = false,  # Deploy only on master/tag, not on PR
)
