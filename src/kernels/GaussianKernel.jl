struct GaussianKernel<:AbstractKernel
    σ::Float64
    σ²::Float64
    λ::Float64
    λ²::Float64

    function GaussianKernel(σ, λ)
        new(σ, σ^2, λ, λ^2)
    end
end

@inline function compute(kernel::GaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64})
    r = x1 - x2
    return kernel.σ² * exp(-dot(r,r)/2kernel.λ²)
end

@inline function compute!(kernel::GaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-dot(r, r)/2kernel.λ²)
end

@inline function ∂K∂σ!(kernel::GaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = 2kernel.σ * exp(-dot(r, r)/2kernel.λ²)
end

@inline function ∂K∂λ!(kernel::GaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    r2 = dot(r,r)
    target[idx...] = kernel.σ² * exp(-r2/2kernel.λ²) * (-r2/(kernel.λ²*kernel.λ))
end
