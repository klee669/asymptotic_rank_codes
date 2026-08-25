# Asymptotic rank codes

This repository contains the Julia script and example numerical data for tensor interpolation computations used to obtain asymptotic-rank bounds. 
This is the repo for the paper ``Asymptotic rank bounds: a numerical census'' (https://arxiv.org/abs/1007.1597)

## Included workflow

Open the following files in Julia and execute them from top to bottom, for example with Shift+Enter in VS Code:

1. `01_collect_points.jl` constructs the square sliced tensor system and either
   loads the included monodromy pool or resumes point collection. (for codim 1, running this file is enough to get a lower bound of the degree for the secant variety)
2. `02_interpolate_and_certify.jl` evaluates the projectively normalized
   Bombieri-Weyl basis, selects rows by pivoted QR, and certifies the selected
   roots.
3. `03_certified_rank.jl` constructs the Acb interval interpolation matrix and
   tests nonsingularity with `Arblib.solve!`.

The default configuration in all three scripts is
`(r,a,b,c) = (9,8,4,4)` with interpolation degree `76`. The directory
`r9_844_s9844/` contains the matching system parameters, 32,224 monodromy
solutions, degree-76 screening data, and 3,003 certified roots. Consequently,
file 01 skips monodromy when run against the included data, while files 02 and
03 can recompute their respective stages.

## Caveat
Large JLD2 files are committed directly. The largest file is approximately 66 MB for the r=9 case.
