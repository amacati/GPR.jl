struct GeneralGaussianKernel<:AbstractKernel
    σ::Real
    σ²::Real
    Λ::SVector{S, T} where {S, T}

    function GeneralGaussianKernel(σ::Real, Λ::Vector{<:Real})
        new(σ, σ^2, SVector{length(Λ), eltype(Λ)}(Λ))
    end
end


@inline function compute(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray)
    r = x1 - x2
    return kernel.σ² * exp(-(r'*(kernel.Λ.*r)))  # More efficient than dot with Diagonal
end

@inline function compute!(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple)
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-(r'*(kernel.Λ.*r)))
end

@inline function ∂K∂σ!(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple)
    r = x1 - x2
    target[idx...] = 2kernel.σ * exp(-(r'*(kernel.Λ.*r)))
end

@inline function ∂K∂Λi!(kernel::GeneralGaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer}, idx_Λ::Integer)
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-(r'*(kernel.Λ.*r))) * (-r[idx_Λ]^2)
end