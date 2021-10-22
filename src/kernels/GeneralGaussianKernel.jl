mutable struct GeneralGaussianKernel<:AbstractKernel
    σ::Real
    σ²::Real
    Λ::SVector{S, T} where {S, T}
    nparams::Integer

    _2σ::Real
    _buffer::MVector{3,Float64}

    function GeneralGaussianKernel(σ::Real, Λ::AbstractArray{<:Real})
        new(σ, σ^2, SVector(Λ...), 1+length(Λ), 2σ, zeros(MVector{3}))
    end
end

@inline function compute(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray)
    r = x1 - x2
    kernel._buffer[1] = -r'*(kernel.Λ.*r)  # More efficient than dot with Diagonal
    return kernel.σ² * exp(kernel._buffer[1])  # kernel.σ² * exp(-(r'*Λ*r))
end

@inline function compute!(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple)
    r = x1 - x2
    kernel._buffer[1] = -r'*(kernel.Λ.*r)
    target[idx...] = kernel.σ² * exp(kernel._buffer[1])
end

@inline function ∂K∂σ!(kernel::GeneralGaussianKernel, X::Vector{<:AbstractVector}, target::AbstractMatrix)
    for i in 1:length(X), j in 1:i
        _∂K∂σ!(kernel, X[i], X[j], target, (i,j))
    end
    return Symmetric(target, :L)
end

@inline function _∂K∂σ!(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple)
    r = x1 - x2
    kernel._buffer[1] = -r'*(kernel.Λ.*r)
    target[idx...] = kernel._2σ * exp(kernel._buffer[1])
end

@inline function ∂K∂Λi!(kernel::GeneralGaussianKernel, X::Vector{<:AbstractVector}, target::AbstractMatrix, Λidx::Integer)
    for i in 1:length(X), j in 1:i
        _∂K∂Λi!(kernel, X[i], X[j], target, (i,j), Λidx)
    end
    return Symmetric(target, :L)
end

@inline function _∂K∂Λi!(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer}, Λidx::Integer)
    r = x1 - x2
    kernel._buffer[1] = -(r'*(kernel.Λ.*r))
    kernel._buffer[2] = -r[Λidx]^2
    kernel._buffer[3] = exp(kernel._buffer[1]) * kernel._buffer[2]
    target[idx...] = kernel.σ² * kernel._buffer[3]  # σ² * exp(-r'*Λ*r) * (-rᵢ²)
end

function get_derivative_handles(kernel::GeneralGaussianKernel)
    ∂K∂Λivec = [_∂K∂Λi!(kernel, X, target) = ∂K∂Λi!(kernel, X, target, Λidx) for Λidx in 1:length(kernel.Λ)]
    return [∂K∂σ!, ∂K∂Λivec...]
end

# Modify kernel in place to avoid creating new kernel objects in optimizations.
function modifykernel!(kernel::GeneralGaussianKernel, param::AbstractArray)
    @assert length(param) == kernel.nparams ("param vector has wrong number of parameters!")
    kernel.σ = param[1]
    kernel.σ² = param[1]^2
    kernel.Λ = SVector(param[2:end]...)
    kernel._2σ = 2param[1]
    return kernel
end

function getparams(kernel::GeneralGaussianKernel)
    return [kernel.σ, kernel.Λ...]
end

function Base.copy(s::GeneralGaussianKernel)
    return GeneralGaussianKernel(s.σ, s.Λ)
end