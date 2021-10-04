struct QuaternionKernel<:AbstractKernel
    σ::Real
    σ²::Real
    λ::Real
    λ²::Real

    function GaussianKernel(σ::Real, λ::Real)
        new(σ, σ^2, λ, λ^2)
    end
end

@inline function compute(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray)
    r = acos(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z + x2.z)
    return kernel.σ² * exp(-r^2/2kernel.λ²)
end

@inline function compute!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = acos(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z + x2.z)
    target[idx...] = kernel.σ² * exp(-r^2/2kernel.λ²)
end

@inline function ∂K∂σ!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = acos(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z + x2.z)
    target[idx...] = 2kernel.σ * exp(-r^2/2kernel.λ²)
end

@inline function ∂K∂λ!(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = acos(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z + x2.z)
    r2 = r^2
    target[idx...] = kernel.σ² * exp(-r2/2kernel.λ²) * (r2/(kernel.λ^3))
end
