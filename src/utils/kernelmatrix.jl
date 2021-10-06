function compute_kernelmatrix(x1::AbstractMatrix, x2::AbstractMatrix, kernel::AbstractKernel)
    kstar = Matrix{Float64}(undef, size(x1, 2), size(x2, 2))
    @inbounds compute_kernelmatrix!(x1, x2, kernel, kstar)
    return kstar
end

function compute_kernelmatrix(x1::Vector, x2::Vector, kernel::AbstractKernel)
    kstar = Matrix{Float64}(undef, length(x1), length(x2))
    @inbounds compute_kernelmatrix!(x1, x2, kernel, kstar)
    return kstar
end

function compute_kernelmatrix!(x1::AbstractMatrix, x2::AbstractMatrix, kernel::AbstractKernel, target::AbstractMatrix)
    for i in 1:size(x1, 2), j in 1:size(x2, 2)
        compute!(kernel, x1[:,i], x2[:,j], target, (i,j))
    end
end

function compute_kernelmatrix!(x1::Vector, x2::Vector, kernel::AbstractKernel, target::AbstractMatrix)
    for i in 1:size(x1, 1), j in 1:size(x2,1)
        compute!(kernel, x1[i], x2[j], target, (i,j))
    end
end 

# For kernel matrices between data set elements, the symmetry of the problem can be used to avoid unnecessary computations.
function compute_kernelmatrix(x::AbstractMatrix, kernel::AbstractKernel)
    k = Matrix{Float64}(undef, size(x,2), size(x,2))
    @inbounds ksymm = compute_kernelmatrix!(x, kernel, k)
    return ksymm
end

function compute_kernelmatrix(x::Vector, kernel::AbstractKernel)
    k = Matrix{Float64}(undef, length(x), length(x))
    @inbounds ksymm = compute_kernelmatrix!(x, kernel, k)
    return ksymm
end

function compute_kernelmatrix!(x::AbstractMatrix, kernel::AbstractKernel, target::AbstractMatrix)
    for i in 1:size(x, 2), j in 1:i
        compute!(kernel, x[:,i], x[:,j], target, (i,j))
    end
    return Symmetric(target, :L)
end

function compute_kernelmatrix!(x::Vector, kernel::AbstractKernel, target::AbstractMatrix)
    for i in 1:size(x, 1), j in 1:i
        compute!(kernel, x[i], x[j], target, (i,j))
    end
    return Symmetric(target, :L)
end

# If only the diagonal is needed, computation can be reduced further
function compute_kerneldiagonal(x::AbstractMatrix, kernel::AbstractKernel)
    σ = Matrix{Float64}(undef, size(x, 2), 1)
    @inbounds compute_kerneldiagonal!(x, kernel, σ)
    return σ
end

function compute_kerneldiagonal(x::Vector, kernel::AbstractKernel)
    σ = Matrix{Float64}(undef, length(x), 1)
    @inbounds compute_kerneldiagonal!(x, kernel, σ)
    return σ
end

function compute_kerneldiagonal!(x::AbstractMatrix, kernel::AbstractKernel, target::AbstractMatrix)
    for i in 1:size(x, 2)
        compute!(kernel, x[:,i], x[:, i], target, (i, 1))
    end
end

function compute_kerneldiagonal!(x::Vector, kernel::AbstractKernel, target::AbstractMatrix)
    for i in 1:length(x)
        compute!(kernel, x[i], x[i], target, (i, 1))
    end
end