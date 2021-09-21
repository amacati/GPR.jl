using LinearAlgebra
include("AbstractKernel.jl")


struct MaternKernel{Nv}<:AbstractKernel
    v::Float64
    l::Float64
    function MaternKernel{Nv}(v, l) where Nv
        @assert Nv in [1.5, 2.5]
        new{Nv}(v, l)
    end
end

function compute(kernel::MaternKernel{1.5}, x1::AbstractArray{Float64}, x2::AbstractArray{Float64})
    r = norm(x1 - x2)
    return (1 + sqrt(3)r/kernel.l) * exp(-sqrt(3)r/kernel.l)
end

function compute!(kernel::MaternKernel{1.5}, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = norm(x1 - x2)
    target[idx...] = (1 + sqrt(3)r/kernel.l) * exp(-sqrt(3)r/kernel.l)
end

function compute(kernel::MaternKernel{2.5}, x1::AbstractArray{Float64}, x2::AbstractArray{Float64})
    r = norm(x1 - x2)
    return (1 + sqrt(3)r/kernel.l) * exp(-sqrt(3)r/kernel.l)
end

function compute!(kernel::MaternKernel{2.5}, x1::AbstractArray{Float64}, x2::AbstractArray{Float64}, target::AbstractArray{Float64}, idx::Tuple{Integer, Integer})
    r = norm(x1 - x2)
    target[idx...] = (1 + sqrt(5)r/kernel.l + 5r^2/(3kernel.l^2)) * exp(-sqrt(5)r/kernel.l)
end
