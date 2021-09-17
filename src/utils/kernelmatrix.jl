using LinearAlgebra
include("../kernels/AbstractKernel.jl")


function compute_kernelmatrix(x1::Matrix{Float64}, x2::Matrix{Float64}, kernel::AbstractKernel)
    kstar = Matrix{Float64}(undef, size(x1, 2), size(x2, 2))
    for i in 1:size(x1, 2), j = 1:size(x2, 2)
        compute!(kernel, x1[:,i], x2[:,j], kstar, (i,j))
    end
    return kstar
end

function compute_kernelmatrix!(x1::Matrix{Float64}, x2::Matrix{Float64}, kernel::AbstractKernel, target::Matrix{Float64})
    for i in 1:size(x1, 2), j = 1:size(x2, 2)
        compute!(kernel, x1[:,i], x2[:,j], target, (i,j))
    end
end

# For kernel matrices of a single vector, the symmetry of the problem can be used to avoid unnecessary computations.
function compute_kernelmatrix(x::Matrix{Float64}, kernel::AbstractKernel)
    k = Matrix{Float64}(undef, size(x), size(x))
    for i in 1:size(x1, 2), j = 1:i
        compute!(kernel, x[:,i], x[:,j], k, (i,j))
    end
    return Symmetric(k, :L)
end

function compute_kernelmatrix!(x::Matrix{Float64}, kernel::AbstractKernel, target::Matrix{Float64})
    for i in 1:size(x, 2), j = 1:i
        compute!(kernel, x[:,i], x[:,j], target, (i,j))
    end
    target = Symmetric(target, :L)
end

function compute_kerneldiagonal(x::Matrix{Float64}, kernel::AbstractKernel)
    σ = Matrix{Float64}(undef, size(x2, 2), 1)
    for i in 1:size(x, 2)
        compute!(kernel, x[:,i], x[:, i], σ, (i, 1))
    end
    return σ
end

function compute_kerneldiagonal!(x::Matrix{Float64}, kernel::AbstractKernel, target::Matrix{Float64})
    for i in 1:size(x, 2)
        compute!(kernel, x[:,i], x[:, i], target, (i, 1))
    end
end