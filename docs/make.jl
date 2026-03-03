# Headless mode for Plots.jl (prevents Qt GUI errors during doc build)
ENV["GKSwstype"] = "100"

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
        return  # Do nothing if the contents are the same
    end
    write(path, content)
end

"""
Copy file only if content changed (prevents mtime update triggering rebuild).
"""
function cp_if_changed(src::String, dst::String)
    if isfile(dst) && read(src) == read(dst)
        return  # Do nothing if the contents are the same
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

    # Remove benchmark markers (invisible in GitHub, but Documenter escapes them)
    content = replace(content, r"<!-- BENCHMARK_[A-Z_]+ -->" => "")

    return content
end

# ============================================
# Helper: Inject Google site verification tag
# ============================================

"""
Inject Google Search Console verification meta tag into generated HTML files.
This is enabled only when `ENV["GOOGLE_SITE_VERIFICATION"]` is set.
"""
function inject_google_site_verification!(build_dir::String)
    token = strip(get(ENV, "GOOGLE_SITE_VERIFICATION", ""))
    isempty(token) && return

    safe_token = replace(token, '"' => "&quot;")
    meta_tag = "<meta name=\"google-site-verification\" content=\"$(safe_token)\" />"
    injected = 0

    for (root, _, files) in walkdir(build_dir)
        for file in files
            endswith(file, ".html") || continue
            path = joinpath(root, file)
            html = read(path, String)
            occursin("google-site-verification", html) && continue
            occursin("</head>", html) || continue

            write_if_changed(path, replace(html, "</head>" => "$(meta_tag)\n</head>"; count=1))
            injected += 1
        end
    end

    @info "Injected google-site-verification meta tag" files=injected build_dir=build_dir
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
mkpath(joinpath(DOCS_SRC, "boundary-conditions"))
mkpath(joinpath(DOCS_SRC, "nd"))

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
    # Disable git remote detection — servedocs() runs from docs/ which breaks
    # git root detection. Source links are handled by deploydocs() in CI.
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://projecttorreypines.github.io/FastInterpolations.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API Selection Guide" => "guides/api_selection.md",
        "Interpolation" => [
            "Overview" => "interpolation/overview.md",
            "Constant" => "interpolation/constant.md",
            "Linear" => "interpolation/linear.md",
            "Quadratic" => "interpolation/quadratic.md",
            "Cubic" => "interpolation/cubic.md",
            "Derivatives" => "interpolation/derivatives.md",
            "Integration" => "interpolation/integration.md",
            "Visual Comparison" => "interpolation/comparison.md",
        ],
        "Boundary Conditions" => [
            "Overview" => "boundary-conditions/overview.md",
            "PointBC & PolyFit" => "boundary-conditions/pointbc.md",
            "PeriodicBC" => "boundary-conditions/periodicbc.md",
        ],
        "Extrapolation" => "extrapolation.md",
        "Visualization" => "visualization.md",
        "Multi-Dimensional Interpolation" => [
            "Overview" => "nd/overview.md",
            "Boundary Conditions" => "nd/boundary_conditions.md",
            "Derivatives" => "nd/derivatives.md",
            "Integration" => "nd/integration.md",
            "Extrapolation" => "nd/extrapolation.md",
        ],
        "Factory Functions" => "guides/factory_functions.md",
        "Advanced Usage" => [
            "Overview" => "guides/advanced_overview.md",
            "Complex Numbers" => "guides/complex_number_support.md",
            "Autodiff (AD)" => [
                "1D Interpolants" => "guides/autodiff_support.md",
                "ND Interpolants" => "guides/autodiff_nd.md",
            ],
            "Optimization (Optim.jl)" => "guides/optimization.md",
            "Search & Hints" => [
                "Overview" => "guides/search/overview.md",
                "Search Policies" => "guides/search/policies.md",
                "Using Hints" => "guides/search/hints.md",
            ],
            "Series Interpolant" => "guides/series_interpolant.md",
            "Memory & Allocation" => "guides/memory_allocation.md",
            "Custom Value Types (Duck Typing)" => "guides/custom_value_types.md",
        ],
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
        "Migration Guides" => [
            "v0.2 → v0.3" => "migration/to_v0.3.md",
        ],
    ],
    doctest = true,
    checkdocs = :exports,
)

inject_google_site_verification!(joinpath(@__DIR__, "build"))

deploydocs(
    repo = "github.com/ProjectTorreyPines/FastInterpolations.jl.git",
    devbranch = "master",
    push_preview = true,  # Deploy only on master/tag, not on PR
)
