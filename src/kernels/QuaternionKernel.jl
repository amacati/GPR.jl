mutable struct QuaternionKernel<:AbstractKernel
    σ::Real
    σ²::Real
    Λ::SVector{S, T} where {S, T}
    nparams::Integer

    _2σ::Real

    function QuaternionKernel(σ::Real, Λ::AbstractArray{<:Real})
        @assert length(Λ) == 3 ("Quaternion Kernel Λ has to be of length 3")
        new(σ, σ^2, SVector(Λ...), 4, 2σ)
    end
end

@inline function compute(kernel::QuaternionKernel, x1::AbstractArray, x2::AbstractArray)
    q1 = UnitQuaternion(x1[1], x1[2:4])
    q2 = UnitQuaternion(x2[1], x2[2:4])
    qr = q2 / q1
    r = [qr.x, qr.y, qr.z]
    return kernel.σ² * exp(-r'*(kernel.Λ.*r))
end

@inline function compute!(kernel::QuaternionKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    q1 = UnitQuaternion(x1[1], x1[2:4])
    q2 = UnitQuaternion(x2[1], x2[2:4])
    qr = q2 / q1
    r = [qr.x, qr.y, qr.z]
    target[idx...] = kernel.σ² * exp(-r'*(kernel.Λ.*r))
end

@inline function ∂K∂σ!(kernel::QuaternionKernel, X::Vector{<:AbstractVector}, target::AbstractMatrix)
    for i in 1:length(X), j in 1:i
        _∂K∂σ!(kernel, X[i], X[j], target, (i,j))
    end
    return Symmetric(target, :L)
end

@inline function _∂K∂σ!(kernel::QuaternionKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer})
    q1 = UnitQuaternion(x1[1], x1[2:4])
    q2 = UnitQuaternion(x2[1], x2[2:4])
    qr = q2 / q1
    r = [qr.x, qr.y, qr.z]
    target[idx...] = kernel._2σ * exp(-r'*(kernel.Λ.*r))
end

@inline function ∂K∂Λi!(kernel::QuaternionKernel, X::Vector{<:AbstractVector}, target::AbstractMatrix, Λidx::Integer)
    for i in 1:length(X), j in 1:i
        _∂K∂Λi!(kernel, X[i], X[j], target, (i,j), Λidx)
    end
    return Symmetric(target, :L)
end

@inline function _∂K∂Λi!(kernel::QuaternionKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractArray, idx::Tuple{Integer, Integer}, Λidx::Integer)
    q1 = UnitQuaternion(x1[1], x1[2:4])
    q2 = UnitQuaternion(x2[1], x2[2:4])
    qr = q2 / q1
    r = [qr.x, qr.y, qr.z]
    target[idx...] = kernel.σ² * exp(-(r'*(kernel.Λ.*r))) * (-r[Λidx]^2)  # σ² * exp(-r'*Λ*r) * (-rᵢ²)
end

function get_derivative_handles(_::QuaternionKernel)
    ∂K∂Λivec = [_∂K∂Λi!(kernel, X, target) = ∂K∂Λi!(kernel, X, target, Λidx) for Λidx in 1:3]
    return [∂K∂σ!, ∂K∂Λivec...]
end

# Modify kernel in place to avoid creating new kernel objects in optimizations.
function modifykernel!(kernel::QuaternionKernel, param::AbstractArray)
    @assert length(param) == kernel.nparams ("param vector has wrong number of parameters!")
    kernel.σ = param[1]
    kernel.σ² = param[1]^2
    kernel.Λ = SVector(param[2:end]...)
    kernel._2σ = 2param[1]
    return kernel
end

function getparams(kernel::QuaternionKernel)
    return [kernel.σ, kernel.Λ...]
end

function Base.copy(s::QuaternionKernel)
    return QuaternionKernel(s.σ, s.Λ)
end