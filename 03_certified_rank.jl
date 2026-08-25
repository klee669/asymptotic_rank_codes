# Run this file from top to bottom with Shift+Enter after file 02.
include(joinpath(@__DIR__, "tensor_interpolation_helpers.jl"))

# Use the precomputed r=9, format (8,4,4) case configured in file 02.
codimension = 2
degree = 76
precision_bits = 1024


case_name = "r9_844_s9844"
output_dir = joinpath(@__DIR__, case_name)
screening_path = joinpath(output_dir, "screening_degree$(degree).jld2")
certified_path = joinpath(
    output_dir,
    "certified_t_degree$(degree)_1-$(binomial(degree + codimension, codimension)).jld2",
)
result_path = joinpath(
    output_dir,
    "highprec_rank_degree$(degree)_$(precision_bits)bit.jld2",
)

# Load certified t-coordinate balls and the selected monomial ordering
n = binomial(degree + codimension, codimension)
certified_t = load(certified_path, "certified_t")
certified_t_radii = load(certified_path, "certified_t_radii")
successful = load(certified_path, "successful")
column_pivots = load(screening_path, "column_pivots")

@show n count(successful) precision_bits
@assert length(certified_t) == n
@assert all(successful)
@assert length(column_pivots) == n

exponents = homogeneous_exponents(degree, codimension + 1)
exponents = exponents[column_pivots]
weights = bombieri_weights_acb(degree, exponents, precision_bits)

matrix = AcbMatrix(n, n; prec=precision_bits)
matrix_started = time()

for row in 1:n
    point = normalized_projective_point_acb(
        certified_t[row],
        certified_t_radii[row],
        precision_bits,
    )
    fill_bombieri_row_acb!(matrix, row, point, degree, exponents, weights)

    if row % 25 == 0 || row == n
        println("built certified row $row / $n " *
                "($(round(time() - matrix_started; digits=1)) s)")
        flush(stdout)
    end
end

# Certified nonsingularity test
rhs = AcbMatrix(n, 1; prec=precision_bits)
solution = similar(rhs)
rhs[1, 1] = 1

solve_started = time()


# This dense 3003-by-3003 interval solve is the expensive final step.
solve_flag = Arblib.solve!(solution, matrix, rhs; prec=precision_bits)

solve_seconds = time() - solve_started
full_rank = !iszero(solve_flag)

@show full_rank solve_seconds

# Save the final result
atomic_jldsave(
    result_path;
    degree,
    codimension,
    precision_bits,
    matrix_size=n,
    full_rank,
    failure=full_rank ? "" : "Arblib.solve! did not certify nonsingularity",
    lu_seconds=solve_seconds,
    total_seconds=(time() - matrix_started),
    certified_path,
    diagnostics_path=screening_path,
    column_pivots,
    use_certified_radii=true,
    factor_method="solve",
)

@show result_path
