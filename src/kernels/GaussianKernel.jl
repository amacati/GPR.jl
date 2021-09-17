using LinearAlgebra
using StaticArrays
include("AbstractKernel.jl")


struct GaussianKernel<:AbstractKernel
    σ::Float64
    l::Float64
end

# TODO: Include general gaussian kernel 
#=
struct GeneralGaussianKernel{N}<:AbstractKernel
    σ::Float64
    M::SMatrix{N,N}

    function GeneralGaussianKernel(σ, M::AbstractVector)
        N = length(M)
        new{N}(σ,Diagonal(M))
    end

    function GeneralGaussianKernel(σ, M::AbstractMatrix)
        N = size(M)[1]
        new{N}(σ, M)
    end
end
=#

function compute(kernel::GaussianKernel, x1::Vector{Float64}, x2::Vector{Float64})::Float64
    r = x1 - x2
    return kernel.σ^2 * exp(-dot(r,r)/2kernel.l^2)
end

function compute(kernel::GaussianKernel, x1::Matrix{Float64}, x2::Matrix{Float64})::Float64
    r = x1 - x2
    return kernel.σ^2 * exp(-dot(r,r)/2kernel.l^2)
end

function compute!(kernel::GaussianKernel, x1::Vector{Float64}, x2::Vector{Float64}, target::Matrix{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = kernel.σ^2 * exp(-dot(r,r)/2kernel.l^2)
end

function compute!(kernel::GaussianKernel, x1::Matrix{Float64}, x2::Matrix{Float64}, target::Matrix{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = kernel.σ^2 * exp(-dot(r, r)/2kernel.l^2)
end
