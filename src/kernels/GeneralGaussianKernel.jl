struct GeneralGaussianKernel<:AbstractKernel
    σ::Float64
    σ²::Float64
    Λ::SVector{S, Float64} where S

    function GeneralGaussianKernel(σ::Real, Λ::Vector{Float64})
        new(σ, σ^2, SVector{length(Λ), Float64}(Λ))
    end
end


@inline function compute(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64})
    r = x1 - x2
    return kernel.σ² * exp(-(r'*(kernel.Λ.*r)))  # More efficient than dot with Diagonal
end

@inline function compute!(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple)
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-(r'*(kernel.Λ.*r)))
end

@inline function ∂K∂σ!(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple)
    r = x1 - x2
    target[idx...] = 2kernel.σ * exp(-(r'*(kernel.Λ.*r)))
end

@inline function ∂K∂Λi!(kernel::GeneralGaussianKernel, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple, idx_Λ::Integer)
    r = x1 - x2
    target[idx...] = kernel.σ² * exp(-(r'*(kernel.Λ.*r))) * (-r[idx_Λ]^2)
end