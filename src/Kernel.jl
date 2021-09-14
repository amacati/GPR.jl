using LinearAlgebra

struct GaussianKernel

    variance::Number
    length::Number

end

function compute(kernel::GaussianKernel, x1::Vector{Float64}, x2::Vector{Float64})::Number
    return kernel.variance^2 * exp(-norm(x1 - x2)^2/2kernel.length^2)
end

function compute(kernel::GaussianKernel, x1::Matrix{Float64}, x2::Matrix{Float64})::Number
    return kernel.variance^2 * exp(-norm(x1 - x2)^2/2kernel.length^2)
end

function compute!(kernel::GaussianKernel, x1::Vector{Float64}, x2::Vector{Float64}, target::Matrix{Float64}, idx::Tuple{Integer, Integer})
    target[idx...] = kernel.variance^2 * exp(-norm(x1 - x2)^2/2kernel.length^2)
end

function compute!(kernel::GaussianKernel, x1::Matrix{Float64}, x2::Matrix{Float64}, target::Matrix{Float64}, idx::Tuple{Integer, Integer})
    target[idx...] = kernel.variance^2 * exp(-norm(x1 - x2)^2/2kernel.length^2)
end