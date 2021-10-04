struct GaussianKernel<:AbstractKernel
    σ::Real
    σ²::Real
    λ::Real
    λ²::Real

    function GaussianKernel(σ::Real, λ::Real)
        new(σ, σ^2, λ, λ^2)
    end
end

@inline function compute(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray)
    r = x1 - x2
    return kernel.σ² * exp(-dot(r,r)/2kernel.λ²)
end

@inline function compute!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-dot(r, r)/2kernel.λ²)
end

@inline function ∂K∂σ!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = 2kernel.σ * exp(-dot(r,r)/2kernel.λ²)
end

@inline function ∂K∂λ!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = x1 - x2
    r2 = dot(r,r)
    target[idx...] = kernel.σ² * exp(-r2/2kernel.λ²) * (r2/(kernel.λ^3))
end
