# Asymptotic rank codes

This repository contains the line-by-line Julia workflow and saved numerical
data for tensor interpolation computations used to obtain asymptotic-rank
bounds. The former implementation is preserved on the
`legacy-pre-line-by-line` branch.

## Included workflow

Open the following files in Julia and execute them from top to bottom, for
example with Shift+Enter in VS Code:

1. `01_collect_points.jl` constructs the square sliced tensor system and either
   loads the included monodromy pool or resumes point collection.
2. `02_interpolate_and_certify.jl` evaluates the projectively normalized
   Bombieri--Weyl basis, selects rows by pivoted QR, and certifies the selected
   roots.
3. `03_certified_rank.jl` constructs the Acb interval interpolation matrix and
   tests nonsingularity with `Arblib.solve!`.

Instantiate the environment once before running the files:

```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
```

The default configuration in all three scripts is
`(r,a,b,c) = (9,8,4,4)` with interpolation degree `76`. The directory
`r9_844_s9844/` contains the matching system parameters, 32,224 monodromy
solutions, degree-76 screening data, and 3,003 certified roots. Consequently,
file 01 skips monodromy when run against the included data, while files 02 and
03 can recompute their respective stages.

The `palmetto_rank_results/` subdirectory records the completed degree-100 and
degree-110 rank computations for the same tensor case.

Large JLD2 files are committed directly. The largest file is approximately
66 MB, below GitHub's 100 MB per-file limit.
