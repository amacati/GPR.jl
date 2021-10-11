mutable struct GaussianKernel<:AbstractKernel
    σ::Real
    σ²::Real
    λ::Real
    λ²::Real
    _2σ::Real
    _2λ²::Real
    _buffer::Vector{Float64}  # Used in compute to avoid memory allocation

    function GaussianKernel(σ::Real, λ::Real)
        new(σ, σ^2, λ, λ^2, 2σ, 2λ^2, zeros(5))
    end
end

# Modify kernel in place to avoid creating new kernel objects in optimizations.
function modifykernel(kernel::GaussianKernel, σ::Real, λ::Real)
    kernel.σ = σ
    kernel.σ² = σ^2
    kernel.λ = λ
    kernel.λ² = λ^2
    kernel._2σ = 2σ
    kernel._2λ² = 2λ^2
    return kernel
end

@inline function compute(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray)
    r = x1 - x2
    kernel._buffer[1] = -dot(r,r)/kernel._2λ²
    kernel._buffer[2] = exp(kernel._buffer[1])
    return kernel.σ² * kernel._buffer[2]  # kernel.σ² * exp(-dot(r, r)/2kernel.λ²)
end

@inline function compute!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractMatrix, idx::Tuple{Integer, Integer})
    r = x1 - x2
    kernel._buffer[1] = -dot(r,r)/kernel._2λ²
    kernel._buffer[2] = exp(kernel._buffer[1])
    target[idx...] = kernel.σ² * kernel._buffer[2]  # kernel.σ² * exp(-dot(r, r)/2kernel.λ²)
end

function ∂K∂σ!(kernel::GaussianKernel, X::AbstractMatrix, target::AbstractMatrix)
    for i in 1:size(X,2), j in 1:i
        _∂K∂σ!(kernel, X[:,i], X[:,j], target, (i,j))
    end
    return Symmetric(target, :L)
end

function ∂K∂σ!(kernel::GaussianKernel, X::Vector{SVector{S, T}}, target::AbstractMatrix) where {S,T}
    for i in 1:length(X), j in 1:i
        _∂K∂σ!(kernel, X[i], X[j], target, (i,j))
    end
    return Symmetric(target, :L)
end

@inline function _∂K∂σ!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractMatrix, idx::Tuple{Integer, Integer})
    r = x1 - x2
    kernel._buffer[1] = -dot(r,r)/kernel._2λ²
    kernel._buffer[2] = exp(kernel._buffer[1])
    target[idx...] = kernel._2σ * kernel._buffer[2]  # 2kernel.σ * exp(-dot(r,r)/2kernel.λ²)
end

function ∂K∂λ!(kernel::GaussianKernel, X::AbstractMatrix, target::AbstractMatrix)
    for i in 1:size(X,2), j in 1:i
        _∂K∂λ!(kernel, X[:,i], X[:,j], target, (i,j))
    end
    return Symmetric(target, :L)
end

function ∂K∂λ!(kernel::GaussianKernel, X::Vector{SVector{S, T}}, target::AbstractMatrix) where {S,T}
    for i in 1:length(X), j in 1:i
        _∂K∂λ!(kernel, X[i], X[j], target, (i,j))
    end
    return Symmetric(target, :L)
end

@inline function _∂K∂λ!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractMatrix, idx::Tuple{Integer, Integer})
    r = x1 - x2
    kernel._buffer[1] = dot(r,r)
    kernel._buffer[2] = -kernel._buffer[1]/kernel._2λ²
    kernel._buffer[3] = exp(kernel._buffer[2])
    kernel._buffer[4] =  kernel._buffer[1]/kernel.λ^3
    kernel._buffer[5] = kernel._buffer[3] * kernel._buffer[4]
    target[idx...] = kernel.σ² * kernel._buffer[5]  # kernel.σ² * exp(-r2/2kernel.λ²) * (r2/(kernel.λ^3)), r2 = dot(r,r)
end

function get_derivative_handles(_::GaussianKernel)
    return [∂K∂σ!, ∂K∂λ!]
end

function Base.copy(s::GaussianKernel)
    return GaussianKernel(s.σ, s.λ)
end