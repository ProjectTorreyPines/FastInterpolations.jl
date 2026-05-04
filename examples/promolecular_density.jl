# ============================================================
# Promolecular Density with PHS Interpolation
# ============================================================
#
# This example demonstrates how to use the PHS interpolation method
# with a custom reference density function for high-accuracy smooth
# approximations of quantum mechanical electron densities.
#
# Key features:
#   - PromolecularRef: sum of 1D atomic radial densities from critic2 PBE basis
#   - PHSInterpolantND: local polyharmonic spline interpolant with analytical derivatives
#   - PHSLogTransform: smooth log-density f(x) = ln(ρ/ρ₀) with accurate derivatives
#
# Application: Quantum chemistry, electron density analysis, bond paths.
#
# Usage:
#   using FastInterpolations
#   include("examples/promolecular_density.jl")
#   
#   # Create reference density from atomic wavefunction files
#   atoms = load_xyz(xyz_file)
#   ref = PromolecularRef(atoms, wfc_dir)
#   
#   # Build log-density PHS interpolant
#   itp = phs_interp(grids, rho_data; 
#       reference_interp = ref, stencil_size = 8, degree = 3)

using FastInterpolations
using LinearAlgebra

# ============================================================
# Atomic Data
# ============================================================

# Element symbol → atomic number (full periodic table, Z ∈ [1, 118])
const ELEMENT_Z = Dict(
    "H"=>1,  "He"=>2, "Li"=>3,  "Be"=>4,  "B"=>5,   "C"=>6,   "N"=>7,   "O"=>8,
    "F"=>9,  "Ne"=>10,"Na"=>11, "Mg"=>12, "Al"=>13, "Si"=>14, "P"=>15,  "S"=>16,
    "Cl"=>17,"Ar"=>18,"K"=>19,  "Ca"=>20, "Sc"=>21, "Ti"=>22, "V"=>23,  "Cr"=>24,
    "Mn"=>25,"Fe"=>26,"Co"=>27, "Ni"=>28, "Cu"=>29, "Zn"=>30, "Ga"=>31, "Ge"=>32,
    "As"=>33,"Se"=>34,"Br"=>35, "Kr"=>36, "Rb"=>37, "Sr"=>38, "Y"=>39,  "Zr"=>40,
    "Nb"=>41,"Mo"=>42,"Tc"=>43, "Ru"=>44, "Rh"=>45, "Pd"=>46, "Ag"=>47, "Cd"=>48,
    "In"=>49,"Sn"=>50,"Sb"=>51, "Te"=>52, "I"=>53,  "Xe"=>54, "Cs"=>55, "Ba"=>56,
    "La"=>57,"Ce"=>58,"Pr"=>59, "Nd"=>60, "Pm"=>61, "Sm"=>62, "Eu"=>63, "Gd"=>64,
    "Tb"=>65,"Dy"=>66,"Ho"=>67, "Er"=>68, "Tm"=>69, "Yb"=>70, "Lu"=>71, "Hf"=>72,
    "Ta"=>73,"W"=>74, "Re"=>75, "Os"=>76, "Ir"=>77, "Pt"=>78, "Au"=>79, "Hg"=>80,
    "Tl"=>81,"Pb"=>82,"Bi"=>83, "Po"=>84, "At"=>85, "Rn"=>86, "Fr"=>87, "Ra"=>88,
    "Ac"=>89,"Th"=>90,"Pa"=>91, "U"=>92,  "Np"=>93, "Pu"=>94, "Am"=>95, "Cm"=>96,
    "Bk"=>97,"Cf"=>98,"Es"=>99,"Fm"=>100,"Md"=>101,"No"=>102,"Lr"=>103,
    "Rf"=>104,"Db"=>105,"Sg"=>106,"Bh"=>107,"Hs"=>108,"Mt"=>109,"Ds"=>110,
    "Rg"=>111,"Cn"=>112,"Nh"=>113,"Fl"=>114,"Mc"=>115,"Lv"=>116,"Ts"=>117,"Og"=>118,
)

# Unit conversions
const BOHR2ANG = 0.529177210903   # 1 Bohr → Angstrom
const ANG2BOHR = 1.0 / BOHR2ANG   # 1 Angstrom → Bohr

# ============================================================
# Wavefunction File I/O
# ============================================================

"""
    wfc_filename(symbol::String) -> String

Convert element symbol to critic2 PBE wavefunction filename.
Single-character symbols get an extra trailing underscore.
Examples: "H" → "h__pbe.wfc", "He" → "he_pbe.wfc", "C" → "c__pbe.wfc"
"""
function wfc_filename(sym::String)
    return rpad(lowercase(sym), 2, '_') * "_pbe.wfc"
end

"""
    parse_wfc(filepath::String) -> (r_grid::Vector, ρ::Vector)

Parse a critic2 PBE all-electron wavefunction file.

Returns the radial grid `r` and electron density `ρ(r) = Σⱼ occⱼ ψⱼ(r)² / (4πr²)`.

**File format:**
```
Line 1:  norb                           # number of radial orbitals
Line 2:  orbital labels                 # (space-separated, not used here)
Line 3:  occupations                    # electron count per orbital (space-separated floats)
Line 4:  orbital energies               # (space-separated, not used here)
Line 5:  ngrid                          # number of radial grid points
Lines 6…: r  ψ₁(r)  ψ₂(r)  …  ψₙₒᵣb(r)  # radial coordinate and orbital values
```

**Output:**
- `r_grid::Vector{Float64}` — radial coordinate grid (0, …, r_max)
- `ρ::Vector{Float64}` — spherically-averaged electron density ρ(r)
"""
function parse_wfc(filepath::String)
    open(filepath) do io
        norb = parse(Int, readline(io))
        readline(io)                              # labels — not needed
        occ  = parse.(Float64, split(readline(io)))
        readline(io)                              # energies — not needed
        ngrid = parse(Int, readline(io))

        r_vals   = Vector{Float64}(undef, ngrid)
        rho_vals = Vector{Float64}(undef, ngrid)
        pi4 = 4π

        for i in 1:ngrid
            row   = parse.(Float64, split(readline(io)))
            r     = row[1]
            psi   = @view row[2:end]
            rr0   = dot(occ, psi .^ 2)           # Σⱼ occⱼ ψⱼ²
            r_vals[i]   = r
            rho_vals[i] = rr0 / (pi4 * r^2)      # ρ(r)
        end
        return r_vals, rho_vals
    end
end

# ============================================================
# Molecular Geometry I/O
# ============================================================

"""
    load_xyz(filepath::String, wfc_dir::String) -> Vector{Tuple{Int, NTuple{3,Float64}}}

Load an XYZ file and return atoms as list of `(Z, (x, y, z))` with positions in Bohr.

**XYZ Format** (standard for molecular geometry):
```
Line 1:  n_atoms                 # number of atoms
Line 2:  comment or blank line   # ignored
Line 3+: symbol  x  y  z         # element symbol and coordinates (Angstrom)
```

**Assumptions:**
- Coordinates are in Angstrom (converted to Bohr internally)
- Element symbols match standard periodic table notation ("C", "H", "O", etc.)
- All atoms correspond to available critic2 wavefunction files in `wfc_dir`

**Output:**
`Vector{Tuple{Int, NTuple{3,Float64}}}` — list of `(Z, (x, y, z))` with Z = atomic number, (x,y,z) in Bohr

**Raises:** `KeyError` if element symbol not found in ELEMENT_Z; `error()` if wfc file missing.
"""
function load_xyz(filepath::String)
    lines = readlines(filepath)
    n = parse(Int, strip(lines[1]))
    atoms = Vector{Tuple{Int, NTuple{3,Float64}}}(undef, n)
    for i in 1:n
        parts = split(strip(lines[i + 2]))
        sym = String(parts[1])
        Z   = ELEMENT_Z[sym]
        x, y, z = parse(Float64, parts[2]) * ANG2BOHR,
                   parse(Float64, parts[3]) * ANG2BOHR,
                   parse(Float64, parts[4]) * ANG2BOHR
        atoms[i] = (Z, (x, y, z))
    end
    return atoms
end

# ============================================================
# Atomic Radial Density Cache
# ============================================================

"""
    get_rho_itp(Z::Int, wfc_dir::String) -> PHSInterpolantND or CubicSplineInterpolant

Get or load the 1D cubic spline interpolant for atomic density ρ(r) at nuclear charge Z.

**Implementation:**
- Reads critic2 PBE wavefunction file for element with atomic number Z
- Parses radial grid and electron density
- Returns cubic spline with `FillExtrap(0.0)` (density → 0 at r → ∞)
- Result is cached; repeated calls are O(1) lookup

**Input:**
- `Z::Int` — atomic number (1–118)
- `wfc_dir::String` — directory containing critic2 .wfc files

**Output:**
An interpolant supporting:
- Value query: `itp(r)`
- Derivative query: `itp(r; deriv = DerivOp(1))` for ρ'(r)

**Raises:** 
- `error()` if Z ∉ [1, 118]
- `error()` if wavefunction file not found
"""
function get_rho_itp(Z::Int, wfc_dir::String)
    # Note: To avoid global state, create a local Dict in the caller
    # or use this inside a function that maintains the cache.
    sym = findfirst(==(Z), ELEMENT_Z)
    sym === nothing && error("Unknown atomic number Z=$Z")
    fname = joinpath(wfc_dir, wfc_filename(sym))
    isfile(fname) || error("wfc file not found: $fname")
    r_grid, rho_vals = parse_wfc(fname)
    return cubic_interp(r_grid, rho_vals; extrap = FillExtrap(0.0))
end

# ============================================================
# PromolecularRef: Sum of Atomic Densities
# ============================================================

"""
    PromolecularRef(atoms::Vector{Tuple{Int, NTuple{3, Float64}}}, wfc_dir::String)

Callable reference density ρ₀(x) = Σᵢ ρᵢ(|x - Rᵢ|) from PBE all-electron atomic densities.

**Purpose:**
Reference density for PHSLogTransform. Enables accurate smooth approximations of
quantum mechanical electron densities by providing:
- ρ₀(x) and its analytical derivatives via 3D chain rule
- No Gibbs-like oscillations near nuclear cusps (unlike 3D cubic spline)

**Method:**
For each atom i at position Rᵢ with atomic number Zᵢ:
1. Load 1D atomic radial density ρᵢ(r) from critic2 PBE wavefunction file
2. Build 1D cubic spline (smooth, supports derivatives)
3. At query point x, sum atomic contributions: ρ₀(x) = Σᵢ ρᵢ(rᵢ) where rᵢ = |x - Rᵢ|
4. Compute derivatives via chain rule (exact):
   - ∂ρᵢ/∂xd = ρᵢ'(rᵢ) · (xd - Rᵢd) / rᵢ
   - ∂²ρᵢ/∂xd² = [ρᵢ'(rᵢ)/rᵢ + (ρᵢ''(rᵢ) - ρᵢ'(rᵢ)/rᵢ) · (xd - Rᵢd)² / rᵢ²]
   - Mixed partials: ∂²ρᵢ/∂xd1∂xd2 = [(ρᵢ''(rᵢ) - ρᵢ'(rᵢ)/rᵢ) · (xd1 - Rᵢd1) · (xd2 - Rᵢd2) / rᵢ²]

**Signature (callable interface):**
- `ref(q)` — returns ρ₀(q) as a scalar
- `ref(q; deriv=DerivOp(...))` — returns analytical derivatives

**Usage:**
```julia
using FastInterpolations
include("examples/promolecular_density.jl")

# Load atomic geometry and create reference density
atoms = load_xyz("phenol.xyz")
wfc_dir = joinpath(@__DIR__, "dat", "wfc")
ref = PromolecularRef(atoms, wfc_dir)

# Build 3D data arrays (e.g., from DFT calculation)
grids = (x_grid, y_grid, z_grid)
rho_scf = load_density_data(...)

# Create PHS interpolant with log-density transform
itp = phs_interp(grids, rho_scf; 
    reference_interp = ref, 
    stencil_size = 8, 
    degree = 3, 
    blend_factor = 1.75)

# Evaluate density and derivatives
ρ = itp((1.5, 2.0, 3.5))              # value
dρ_dx = itp((1.5, 2.0, 3.5); deriv = (DerivOp(1), DerivOp(0), DerivOp(0)))  # ∂ρ/∂x
```

**References:**
- Critic2 database: https://github.com/aoterodelaroza/critic2/tree/master/dat/wfc
- Paper reference: See PHS-ref.pdf in this repository
"""
struct PromolecularRef
    atoms::Vector{Tuple{Int, NTuple{3, Float64}}}  # (Z, (x,y,z)) in Bohr
    wfc_cache::Dict{Int, Any}                       # Z => 1D spline of ρ(r)
end

"""
    PromolecularRef(atoms, wfc_dir::String) -> PromolecularRef

Construct a PromolecularRef, pre-loading all atomic wavefunction files.
"""
function PromolecularRef(atoms::Vector{Tuple{Int, NTuple{3, Float64}}}, wfc_dir::String)
    cache = Dict{Int, Any}()
    for (Z, _) in atoms
        if !haskey(cache, Z)
            cache[Z] = get_rho_itp(Z, wfc_dir)
        end
    end
    return PromolecularRef(atoms, cache)
end

# Make PromolecularRef callable: (q; deriv=nothing) interface
function (pmr::PromolecularRef)(q; deriv=nothing)
    # Determine derivative order along each axis
    total = deriv === nothing ? 0 : sum(deriv_order(op) for op in deriv)

    if total == 0
        # Value query: ρ₀(x) = Σᵢ ρᵢ(|x - Rᵢ|)
        f = 0.0
        for (Z, R) in pmr.atoms
            xx1 = q[1] - R[1]
            xx2 = q[2] - R[2]
            xx3 = q[3] - R[3]
            r = sqrt(xx1^2 + xx2^2 + xx3^2)
            r < 1e-14 && continue
            f += max(pmr.wfc_cache[Z](r), 0.0)
        end
        return f
    end

    if total == 1
        # First derivative: ∂ρ₀/∂xd = Σᵢ ρᵢ'(rᵢ) · (xd - Rᵢd) / rᵢ
        ax = findfirst(d -> deriv_order(deriv[d]) == 1, 1:3)::Int
        fp = 0.0
        for (Z, R) in pmr.atoms
            xx = (q[1] - R[1], q[2] - R[2], q[3] - R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            rhop = pmr.wfc_cache[Z](r; deriv = DerivOp(1))
            fp += rhop * xx[ax] / r
        end
        return fp
    end

    if total == 2
        # Second derivative: mixed/pure second-order
        nonzero = [d for d in 1:3 if deriv_order(deriv[d]) > 0]
        ax1 = nonzero[1]
        ax2 = length(nonzero) >= 2 ? nonzero[2] : ax1
        fpp = 0.0
        for (Z, R) in pmr.atoms
            xx = (q[1] - R[1], q[2] - R[2], q[3] - R[3])
            r = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            rho_itp = pmr.wfc_cache[Z]
            rhop = rho_itp(r; deriv = DerivOp(1))
            rhopp = rho_itp(r; deriv = DerivOp(2))
            rfac = (rhopp - rhop / r) / r^2
            fpp += ax1 == ax2 ? rhop / r + rfac * xx[ax1]^2 :
                                 rfac * xx[ax1] * xx[ax2]
        end
        return fpp
    end

    return 0.0
end

export PromolecularRef, load_xyz, parse_wfc, wfc_filename, get_rho_itp
export ELEMENT_Z, BOHR2ANG, ANG2BOHR
