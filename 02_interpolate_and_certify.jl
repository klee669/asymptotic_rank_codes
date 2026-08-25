# Run this file from top to bottom with Shift+Enter after 01_collect_points.jl.
include(joinpath(@__DIR__, "tensor_interpolation_helpers.jl"))

# Precomputed r=9, format (8,4,4) case. File 01 loads the included
# 32,224-point pool without rerunning monodromy.
tensor_rank = 9
a_size = 8
b_size = 4
c_size = 4
codimension = 2

degree = 76
candidate_count = 6000
random_seed = 9844
certification_precision = 1024
maximum_certification_precision = 4096

case_name = "r9_844_s9844"
output_dir = joinpath(@__DIR__, case_name)
solutions_path = joinpath(output_dir, "monodromy_solutions.jld2")
parameter_path = joinpath(output_dir, "p0.jld2")
screening_path = joinpath(output_dir, "screening_degree$(degree).jld2")

# Load the monodromy points
sols = load(solutions_path, "sols")
n_projective_coordinates = codimension + 1
n_monomials = binomial(degree + codimension, codimension)
selected_count = n_monomials

@show length(sols) degree candidate_count selected_count
@assert candidate_count <= length(sols)

# Choose an oversampled candidate set
rng = MersenneTwister(random_seed)
candidate_indices = randperm(rng, length(sols))[1:candidate_count]

# Build the Float64 Bombieri screening matrix
exponents = homogeneous_exponents(degree, n_projective_coordinates)
weights_float64 = bombieri_weights_float64(degree, exponents)
screening_matrix = Matrix{ComplexF64}(undef, candidate_count, n_monomials)

screening_started = time()
for (row, solution_index) in enumerate(candidate_indices)
    point = normalized_projective_point_float64(
        sols[solution_index],
        codimension,
    )
    fill_bombieri_row_float64!(
        view(screening_matrix, row, :),
        point,
        degree,
        exponents,
        weights_float64,
    )
    if row % 250 == 0 || row == candidate_count
        println("built screening row $row / $candidate_count " *
                "($(round(time() - screening_started; digits=1)) s)")
        flush(stdout)
    end
end

row_selection = qr!(copy(adjoint(screening_matrix)), ColumnNorm())
selected_local_rows = row_selection.p[1:selected_count]
selected_indices = candidate_indices[selected_local_rows]

atomic_jldsave(
    screening_path;
    candidate_indices,
    selected_indices,
    exponents,
    random_seed,
    degree,
    codimension,
)

selected_matrix = screening_matrix[selected_local_rows, :]
screening_matrix = nothing
GC.gc()

column_selection = qr!(copy(selected_matrix), ColumnNorm())
qr_diagonal = abs.(diag(column_selection.R))
qr_diagonal_ratio = minimum(qr_diagonal) / maximum(qr_diagonal)
column_pivots = column_selection.p

atomic_jldsave(
    screening_path;
    candidate_indices,
    selected_indices,
    exponents,
    qr_diagonal,
    qr_diagonal_ratio,
    column_pivots,
    random_seed,
    degree,
    codimension,
)

@show qr_diagonal_ratio

# Rebuild the tensor system for root certification
p0 = load(parameter_path, "p0")
source_decomposition = load(parameter_path, "source_decomposition")
fiber_slice_matrix = load(parameter_path, "fiber_slice_matrix")
system, _, _ = build_tensor_problem(
    tensor_rank,
    a_size,
    b_size,
    c_size,
    codimension,
    source_decomposition,
    fiber_slice_matrix,
)
certification_cache = CertificationCache(system)
certification_parameters = CertificationParameters(
    p0;
    prec=certification_precision,
)

certified_t = [Complex{BigFloat}[] for _ in selected_indices]
certified_t_radii = [NTuple{2,BigFloat}[] for _ in selected_indices]
successful = falses(length(selected_indices))

certified_path = joinpath(
    output_dir,
    "certified_t_degree$(degree)_1-$(length(selected_indices)).jld2",
)

for (position, solution_index) in enumerate(selected_indices)
    solution = sols[solution_index]
    try
        certificate = HomotopyContinuation.extended_prec_certify_solution(
            system,
            solution,
            solution,
            certification_parameters,
            certification_cache,
            position,
            false;
            max_precision=maximum_certification_precision,
            extended_certificate=true,
        )
        interval = certified_solution_interval_after_krawczyk(certificate)
        t_interval = vec(interval)[(end - codimension + 1):end]

        certified_t[position] = [
            midpoint_bigfloat(midpoint(value), certification_precision)
            for value in t_interval
        ]
        certified_t_radii[position] = [
            (
                radius_bigfloat(real(value), certification_precision),
                radius_bigfloat(imag(value), certification_precision),
            ) for value in t_interval
        ]
        successful[position] = true
    catch err
        @warn "Certification failed" position solution_index exception=(err, catch_backtrace())
    end

    if position % 25 == 0 || position == length(selected_indices)
        atomic_jldsave(
            certified_path;
            selected_indices,
            certified_t,
            certified_t_radii,
            successful,
            batch_start=1,
            batch_stop=length(selected_indices),
            precision_bits=certification_precision,
            max_precision_bits=maximum_certification_precision,
            codimension,
        )
        println("certified $position / $(length(selected_indices)); " *
                "successful=$(count(successful))")
        flush(stdout)
    end
end

@show count(successful) length(successful) certified_path
