mutable struct GaussianKernel<:AbstractKernel
    σ::Real
    σ²::Real
    λ::Real
    λ²::Real
    nparams::Integer

    _2σ::Real
    _2λ²::Real
    _buffer::MVector{5,Float64}  # Used in compute to avoid memory allocation

    function GaussianKernel(σ::Real, λ::Real)
        new(σ, σ^2, λ, λ^2, 2, 2σ, 2λ^2, zeros(MVector{5}))
    end
end

# Modify kernel in place to avoid creating new kernel objects in optimizations.
function modifykernel!(kernel::GaussianKernel, param::AbstractArray)
    @assert length(param) == kernel.nparams ("param vector has wrong number of parameters!")
    kernel.σ = param[1]
    kernel.σ² = param[1]^2
    kernel.λ = param[2]
    kernel.λ² = param[2]^2
    kernel._2σ = 2param[1]
    kernel._2λ² = 2param[2]^2
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

@inline function ∂K∂σ!(kernel::GaussianKernel, X::AbstractMatrix, target::AbstractMatrix)
    for i in 1:size(X,2), j in 1:i
        _∂K∂σ!(kernel, X[:,i], X[:,j], target, (i,j))
    end
    return Symmetric(target, :L)
end

@inline function ∂K∂σ!(kernel::GaussianKernel, X::Vector{<:AbstractVector}, target::AbstractMatrix)
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

@inline function ∂K∂λ!(kernel::GaussianKernel, X::Vector{<:AbstractVector}, target::AbstractMatrix)
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
    target[idx...] = kernel.σ² * kernel._buffer[5]  # kernel.σ² * exp(-r²/2kernel.λ²) * (r²/(kernel.λ³))
end

function getparams(kernel::GaussianKernel)
    return [kernel.σ, kernel.λ]
end

function get_derivative_handles(_::GaussianKernel)
    return [∂K∂σ!, ∂K∂λ!]
end

function Base.copy(s::GaussianKernel)
    return GaussianKernel(s.σ, s.λ)
end