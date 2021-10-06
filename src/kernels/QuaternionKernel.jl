struct QuaternionKernel<:AbstractKernel
    σ::Real
    σ²::Real
    λ::Real
    λ²::Real

    function QuaternionKernel(σ::Real, λ::Real)
        new(σ, σ^2, λ, λ^2)
    end
end

@inline function compute(kernel::QuaternionKernel, x1::UnitQuaternion, x2::UnitQuaternion)
    r = acos(min(abs(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z * x2.z), 1))
    return kernel.σ² * exp(-r^2/2kernel.λ²)
end

@inline function compute!(kernel::QuaternionKernel, x1::UnitQuaternion, x2::UnitQuaternion, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = acos(min(abs(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z * x2.z), 1))
    target[idx...] = kernel.σ² * exp(-r^2/2kernel.λ²)
end

@inline function ∂K∂σ!(kernel::QuaternionKernel, x1::UnitQuaternion, x2::UnitQuaternion, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = acos(min(abs(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z * x2.z), 1))
    target[idx...] = 2kernel.σ * exp(-r^2/2kernel.λ²)
end

@inline function ∂K∂λ!(kernel::QuaternionKernel, x1::UnitQuaternion, x2::UnitQuaternion, target::AbstractArray, idx::Tuple{Integer, Integer})
    r = acos(min(abs(x1.w * x2.w + x1.x * x2.x + x1.y * x2.y + x1.z * x2.z), 1))
    r2 = r^2
    target[idx...] = kernel.σ² * exp(-r2/2kernel.λ²) * (r2/(kernel.λ^3))
end
