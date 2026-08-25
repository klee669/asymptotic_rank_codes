# Run this file from top to bottom with Shift+Enter.

# Packages and helper functions
include(joinpath(@__DIR__, "tensor_interpolation_helpers.jl"))

# Case parameters. The codimension here is the ACTUAL codimension of the image.
tensor_rank = 9
a_size = 8
b_size = 4
c_size = 4
codimension = 2

target = 32224
random_seed = UInt32(9844)
timeout_seconds = 71.0 * 60 * 60

case_name = "r9_844_s9844"
output_dir = joinpath(@__DIR__, case_name)
solutions_path = joinpath(output_dir, "monodromy_solutions.jld2")
parameter_path = joinpath(output_dir, "p0.jld2")
mkpath(output_dir)

# Parameter, image, and generic-fiber dimensions
tensor_size = a_size * b_size * c_size
decomposition_dimension = tensor_rank * (a_size - 1 + b_size - 1 + c_size)
image_dimension = tensor_size - codimension
fiber_dimension = decomposition_dimension - image_dimension

@show tensor_rank a_size b_size c_size tensor_size
@show decomposition_dimension image_dimension codimension fiber_dimension

resuming = isfile(solutions_path) && isfile(parameter_path)
if resuming
    starts = load(solutions_path, "sols")
    p0 = load(parameter_path, "p0")
    source_decomposition = load(parameter_path, "source_decomposition")
    fiber_slice_matrix = load(parameter_path, "fiber_slice_matrix")
    println("Resuming from $(length(starts)) distinct t-points")
else
    rng = MersenneTwister(random_seed)
    source_decomposition = randn(rng, ComplexF64, decomposition_dimension)
    fiber_slice_matrix = randn(
        rng,
        ComplexF64,
        fiber_dimension,
        decomposition_dimension,
    )
end

system, tensor, decomposition_variables = build_tensor_problem(
    tensor_rank,
    a_size,
    b_size,
    c_size,
    codimension,
    source_decomposition,
    fiber_slice_matrix,
)

@show size(system)
@assert size(system, 1) == size(system, 2)

if !resuming
    x0, p0 = make_start_pair(
        tensor,
        decomposition_variables,
        tensor_size,
        codimension,
        source_decomposition,
        rng,
    )
    starts = [x0]

    atomic_jldsave(
        parameter_path;
        p0,
        source_decomposition,
        fiber_slice_matrix,
        tensor_rank,
        a_size,
        b_size,
        c_size,
        codimension,
        fiber_dimension,
        random_seed,
    )
end

# The repository already contains 32,224 points for this case. Running the
# file therefore validates and loads them without starting another long job.
# If the file is absent or contains fewer points, the same cells resume the
# monodromy computation and save its newest checkpoint.
if length(starts) >= target
    sols = starts
    println("The saved pool already contains the target $target points; " *
            "skipping monodromy.")
else
    result = monodromy_solve(
        system,
        starts,
        p0;
        seed=random_seed,
        target_solutions_count=target,
        min_solutions=target,
        max_loops_no_progress=50,
        timeout=timeout_seconds,
        trace_test=false,
        distance=(x, y) -> t_distance(x, y, codimension),
        triangle_inequality=true,
        reuse_loops=:all,
        check_startsolutions=!resuming,
        threading=Threads.nthreads() > 1,
        show_progress=true,
        compile=false,
    )

    sols = solutions(result)
    atomic_jldsave(
        solutions_path;
        sols,
        tensor_rank,
        a_size,
        b_size,
        c_size,
        codimension,
        fiber_dimension,
        target,
        random_seed,
        returncode=result.returncode,
        distinct_t_points=length(sols),
    )
    @show result.returncode
end

@show length(sols) target
length(sols) >= target || @warn "Run this file again to resume from the saved points"
