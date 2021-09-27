struct GaussianKernel<:AbstractKernel
    σ::Float64
    σ²::Float64
    λ::Float64
    λ²::Float64

    function GaussianKernel(σ, λ)
        new(σ, σ^2, λ, λ^2)
    end
end

struct GeneralGaussianKernel<:AbstractKernel
    σ::Float64
    σ²::Float64
    Λ::SVector{Float64}

    function GeneralGaussianKernel(σ, Λ::Vector{Float64})
        new(σ, σ^2, inv.(Λ))
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

@inline function compute(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64})
    r = x1 - x2
    return kernel.σ² * exp(-(r'*(kernel.Λ.*r)))  # More efficient than dot with Diagonal
end

@inline function compute!(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-(r'*(kernel.Λ.*r)))
end

@inline function ∂K∂σ!(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = x1 - x2
    target[idx...] = 2kernel.σ * exp(-(r'*(kernel.Λ.*r)))
end

@inline function ∂K∂Λi!(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer}, idx_Λ::Integer)
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-(r'*(kernel.Λ.*r))) * (-r[idx_Λ]^2/kernel.Λ[idx_Λ]^2)
end