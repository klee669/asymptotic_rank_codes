using HomotopyContinuation
using Arblib
using JLD2
using LinearAlgebra
using Random

import HomotopyContinuation: CertificationCache, CertificationParameters

function atomic_jldsave(path; kwargs...)
    mkpath(dirname(path))
    temporary_path = path * ".tmp"
    jldsave(temporary_path; kwargs...)
    mv(temporary_path, path; force=true)
    path
end

function build_tensor_problem(
    tensor_rank,
    a_size,
    b_size,
    c_size,
    codimension,
    source_decomposition,
    fiber_slice_matrix,
)
    tensor_size = a_size * b_size * c_size
    decomposition_dimension = tensor_rank * (
        a_size - 1 + b_size - 1 + c_size
    )
    image_dimension = tensor_size - codimension
    fiber_dimension = decomposition_dimension - image_dimension

    codimension >= 1 || error("codimension must be positive")
    fiber_dimension >= 0 || error(
        "The supplied codimension gives negative fiber dimension $fiber_dimension",
    )
    length(source_decomposition) == decomposition_dimension || error(
        "source_decomposition must have length $decomposition_dimension",
    )
    size(fiber_slice_matrix) == (fiber_dimension, decomposition_dimension) || error(
        "fiber_slice_matrix must have size " *
        "($fiber_dimension, $decomposition_dimension)",
    )

    @var a[1:tensor_rank, 1:(a_size - 1)]
    @var b[1:tensor_rank, 1:(b_size - 1)]
    @var c[1:tensor_rank, 1:c_size]

    @var A[1:tensor_size, 1:codimension]
    @var B[1:tensor_size]
    @var t[1:codimension]

    tensor = sum(
        kron(c[i, :], [b[i, :]; 1], [a[i, :]; 1]) for i in 1:tensor_rank
    )
    decomposition_variables = vec([a b c])
    variables = [decomposition_variables; t]
    parameters = vec([A B])
    tensor_equations = tensor - (A * t + B)
    fiber_slices = fiber_slice_matrix * decomposition_variables -
                   fiber_slice_matrix * source_decomposition
    equations = [tensor_equations; fiber_slices]
    system = InterpretedSystem(System(equations; variables, parameters))

    size(system, 1) == size(system, 2) || error(
        "Internal error: sliced tensor system is not square: $(size(system))",
    )

    system, tensor, decomposition_variables
end

function make_start_pair(
    tensor,
    decomposition_variables,
    tensor_size,
    codimension,
    source_decomposition,
    rng,
)
    t = randn(rng, ComplexF64, codimension)
    A = randn(rng, ComplexF64, tensor_size, codimension)
    tensor_value = evaluate(
        tensor,
        decomposition_variables => source_decomposition,
    )
    B = tensor_value - A * t
    x0 = [source_decomposition; t]
    p0 = vec([A B])
    x0, p0
end

# Quotient the permutations of rank-one summands without enumerating S_rank.
# Two decompositions represent the same interpolation point exactly when their
# final codimension plane coordinates agree.
t_distance(x, y, codimension) = norm(
    view(x, (length(x) - codimension + 1):length(x)) .-
    view(y, (length(y) - codimension + 1):length(y)),
    Inf,
)

function homogeneous_exponents(degree, number_of_variables)
    exponents = Vector{Vector{Int}}()
    sizehint!(exponents, binomial(degree + number_of_variables - 1, number_of_variables - 1))

    function append_exponents!(prefix, remaining_degree, remaining_variables)
        if remaining_variables == 1
            push!(exponents, [prefix; remaining_degree])
            return
        end
        for exponent in 0:remaining_degree
            append_exponents!(
                [prefix; exponent],
                remaining_degree - exponent,
                remaining_variables - 1,
            )
        end
    end

    append_exponents!(Int[], degree, number_of_variables)
    exponents
end

function bombieri_weights_float64(degree, exponents)
    numerator = factorial(big(degree))
    [
        sqrt(Float64(
            numerator ÷ prod(factorial(big(exponent)) for exponent in exponent_vector),
        )) for exponent_vector in exponents
    ]
end

function normalized_projective_point_float64(solution, codimension)
    point = ComplexF64[solution[(end - codimension + 1):end]; 1]
    point ./ norm(point)
end

function fill_bombieri_row_float64!(row, point, degree, exponents, weights)
    powers = [Vector{ComplexF64}(undef, degree + 1) for _ in eachindex(point)]

    @inbounds for variable in eachindex(point)
        powers[variable][1] = 1
        for k in 1:degree
            powers[variable][k + 1] = powers[variable][k] * point[variable]
        end
    end

    @inbounds for column in eachindex(exponents)
        row[column] = weights[column] * prod(
            powers[variable][exponents[column][variable] + 1]
            for variable in eachindex(point)
        )
    end
    row ./= norm(row)
    row
end

function midpoint_bigfloat(value, precision_bits)
    setprecision(BigFloat, precision_bits) do
        Complex{BigFloat}(BigFloat(real(value)), BigFloat(imag(value)))
    end
end

function radius_bigfloat(value, precision_bits)
    setprecision(BigFloat, precision_bits) do
        BigFloat(radius(Arf, value))
    end
end

function enclosing_arb(midpoint_value, radius_value, precision_bits)
    working_precision = max(
        precision(midpoint_value),
        precision(radius_value),
        precision_bits,
    ) + 32
    lower, upper = setprecision(BigFloat, working_precision) do
        midpoint_big = BigFloat(midpoint_value)
        radius_big = BigFloat(radius_value)
        lower = setrounding(BigFloat, RoundDown) do
            midpoint_big - radius_big
        end
        upper = setrounding(BigFloat, RoundUp) do
            midpoint_big + radius_big
        end
        lower, upper
    end
    Arb((lower, upper); prec=precision_bits)
end

function enclosing_acb(midpoint_value, radii, precision_bits)
    real_ball = enclosing_arb(real(midpoint_value), radii[1], precision_bits)
    imaginary_ball = enclosing_arb(imag(midpoint_value), radii[2], precision_bits)
    Acb(real_ball, imaginary_ball; prec=precision_bits)
end

function normalized_projective_point_acb(midpoints, radii, precision_bits)
    point = [
        enclosing_acb(midpoints[index], radii[index], precision_bits)
        for index in eachindex(midpoints)
    ]
    push!(point, Acb(1; prec=precision_bits))
    projective_norm = sqrt(sum(abs2, point))
    Arblib.contains_zero(projective_norm) && error("Projective norm contains zero")
    [value / projective_norm for value in point]
end

function bombieri_weights_acb(degree, exponents, precision_bits)
    numerator = factorial(big(degree))
    [
        begin
            multinomial = numerator ÷ prod(
                factorial(big(exponent)) for exponent in exponent_vector
            )
            sqrt(Arb(multinomial; prec=precision_bits))
        end for exponent_vector in exponents
    ]
end

function fill_bombieri_row_acb!(matrix, row, point, degree, exponents, weights)
    powers = [Vector{Acb}(undef, degree + 1) for _ in eachindex(point)]

    @inbounds for variable in eachindex(point)
        powers[variable][1] = Acb(1; prec=precision(matrix))
        for k in 1:degree
            powers[variable][k + 1] = powers[variable][k] * point[variable]
        end
    end

    @inbounds for column in eachindex(exponents)
        matrix[row, column] = weights[column] * prod(
            powers[variable][exponents[column][variable] + 1]
            for variable in eachindex(point)
        )
    end
    matrix
end
