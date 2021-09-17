using LinearAlgebra
include("AbstractKernel.jl")


struct GaussianKernel<:AbstractKernel
    σ::Float64
    l::Float64
end

struct GeneralGaussianKernel<:AbstractKernel
    σ::Float64
    M::Matrix{Float64}

    function GeneralGaussianKernel(σ, M::AbstractVector)
        N = length(M)
        new(σ,inv(Diagonal(M)))
    end

    function GeneralGaussianKernel(σ, M::AbstractMatrix)
        new(σ, inv(M))
    end
end


function compute(kernel::GaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64})
    r = x1 - x2
    return kernel.σ^2 * exp(-dot(r,r)/2kernel.l^2)
end

function compute!(kernel::GaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = kernel.σ^2 * exp(-dot(r, r)/2kernel.l^2)
end

function compute(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64})
    r = x1 - x2
    return kernel.σ^2 * exp(-dot(r, kernel.M, r))
end

function compute!(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = kernel.σ^2 * exp(-dot(r, kernel.M, r))
end
