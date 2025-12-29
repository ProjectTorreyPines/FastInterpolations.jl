using Documenter
using FastInterpolations
using Literate

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
# Step 1: Auto-generate content (DRY principle)
# ============================================

const DOCS_SRC = joinpath(@__DIR__, "src")
const EXAMPLES_DIR = joinpath(@__DIR__, "../examples")
const TUTORIALS_DIR = joinpath(DOCS_SRC, "tutorials")

# Create directories (may not exist initially)
mkpath(DOCS_SRC)
mkpath(joinpath(DOCS_SRC, "guides"))
mkpath(TUTORIALS_DIR)

# 1a. Copy images directory
const IMAGES_SRC = joinpath(@__DIR__, "images")
const IMAGES_DST = joinpath(DOCS_SRC, "images")
if isdir(IMAGES_SRC)
    mkpath(IMAGES_DST)
    for img in readdir(IMAGES_SRC)
        src = joinpath(IMAGES_SRC, img)
        dst = joinpath(IMAGES_DST, img)
        if isfile(src)
            cp(src, dst; force=true)
        end
    end
end

# 1b. Copy README.md → index.md (with path rewriting)
readme_content = read(joinpath(@__DIR__, "../README.md"), String)
write(joinpath(DOCS_SRC, "index.md"), rewrite_readme_paths(readme_content; from_root=true))

# 1c. Copy benchmark/README.md → guides/performance.md (with path rewriting)
bench_readme = joinpath(@__DIR__, "../benchmark/README.md")
if isfile(bench_readme)
    bench_content = read(bench_readme, String)
    write(joinpath(DOCS_SRC, "guides/performance.md"),
          rewrite_readme_paths(bench_content; from_root=false))
end

# 1d. Generate tutorials from examples/*.jl using Literate.jl
if isdir(EXAMPLES_DIR)
    for file in sort(readdir(EXAMPLES_DIR))  # Sort for deterministic order
        if endswith(file, ".jl")
            Literate.markdown(
                joinpath(EXAMPLES_DIR, file),
                TUTORIALS_DIR;
                documenter = true,
                credit = false
            )
        end
    end
end

# ============================================
# Step 2: Build documentation
# ============================================

# Collect generated tutorial pages (sorted order)
tutorial_pages = if isdir(EXAMPLES_DIR) && !isempty(readdir(EXAMPLES_DIR))
    sort(["tutorials/$(replace(f, ".jl" => ".md"))"
          for f in readdir(EXAMPLES_DIR) if endswith(f, ".jl")])
else
    String[]
end

makedocs(
    sitename = "FastInterpolations.jl",
    authors = "Min-Gu Yoo",
    modules = [FastInterpolations],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://projecttorreypines.github.io/FastInterpolations.jl",
        assets = String[],
    ),
    pages = vcat(
        ["Home" => "index.md"],
        isempty(tutorial_pages) ? [] : ["Tutorials" => tutorial_pages],
        [
            "Guides" => [
                "Performance" => "guides/performance.md",
            ],
            "API Reference" => [
                "Linear" => "api/linear.md",
                "Cubic" => "api/cubic.md",
                "Types" => "api/types.md",
            ],
            "Internals" => "internals.md",
        ]
    ),
    doctest = true,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/ProjectTorreyPines/FastInterpolations.jl.git",
    devbranch = "master",
    push_preview = true,
)
