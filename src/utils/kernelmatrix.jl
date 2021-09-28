function compute_kernelmatrix(x1::Matrix{Float64}, x2::Matrix{Float64}, kernel::AbstractKernel)
    kstar = Matrix{Float64}(undef, size(x1, 2), size(x2, 2))
    @inbounds compute_kernelmatrix!(x1, x2, kernel, kstar)
    return kstar
end

function compute_kernelmatrix(x1::Vector{SVector{S1,Float64}}, x2::Vector{SVector{S2,Float64}}, kernel::AbstractKernel) where {S1, S2}
    kstar = Matrix{Float64}(undef, length(x1), length(x2))
    @inbounds compute_kernelmatrix!(x1, x2, kernel, kstar)
    return kstar
end

function compute_kernelmatrix!(x1::Matrix{Float64}, x2::Matrix{Float64}, kernel::AbstractKernel, target::Matrix{Float64})
    for i in 1:size(x1, 2), j in 1:size(x2, 2)
        compute!(kernel, x1[:,i], x2[:,j], target, (i,j))
    end
end

function compute_kernelmatrix!(x1::Vector{SVector{S1,Float64}}, x2::Vector{SVector{S2,Float64}}, kernel::AbstractKernel, target::Matrix{Float64}) where {S1, S2}
    for i in 1:size(x1, 1), j in 1:size(x2,1)
        compute!(kernel, x1[i], x2[j], target, (i,j))
    end
end 

# For kernel matrices of a single vector, the symmetry of the problem can be used to avoid unnecessary computations.
function compute_kernelmatrix(x::Matrix{Float64}, kernel::AbstractKernel)
    k = Matrix{Float64}(undef, size(x,2), size(x,2))
    @inbounds compute_kernelmatrix!(x, kernel, k)
    return k
end

function compute_kernelmatrix(x::Vector{SVector{S,Float64}}, kernel::AbstractKernel) where S
    k = Matrix{Float64}(undef, length(x), length(x))
    @inbounds compute_kernelmatrix!(x, kernel, k)
    return k
end

function compute_kernelmatrix!(x::Matrix{Float64}, kernel::AbstractKernel, target::Matrix{Float64})
    for i in 1:size(x, 2), j in 1:i
        compute!(kernel, x[:,i], x[:,j], target, (i,j))
    end
    target = Symmetric(target, :L)
end

function compute_kernelmatrix!(x::Vector{SVector{S,Float64}}, kernel::AbstractKernel, target::Matrix{Float64}) where S
    for i in 1:size(x, 1), j in 1:i
        compute!(kernel, x[i], x[j], target, (i,j))
    end
    target = Symmetric(target, :L)
end

# If only the diagonal is needed, computation can be reduced further
function compute_kerneldiagonal(x::Matrix{Float64}, kernel::AbstractKernel)
    σ = Matrix{Float64}(undef, size(x, 2), 1)
    @inbounds compute_kerneldiagonal!(x, kernel, σ)
    return σ
end

function compute_kerneldiagonal(x::Vector{SVector{S,Float64}}, kernel::AbstractKernel) where S
    σ = Matrix{Float64}(undef, length(x), 1)
    @inbounds compute_kerneldiagonal!(x, kernel, σ)
    return σ
end

function compute_kerneldiagonal!(x::Matrix{Float64}, kernel::AbstractKernel, target::Matrix{Float64})
    for i in 1:size(x, 2)
        compute!(kernel, x[:,i], x[:, i], target, (i, 1))
    end
end

function compute_kerneldiagonal!(x::Vector{SVector{S,Float64}}, kernel::AbstractKernel, target::Matrix{Float64}) where S
    for i in 1:length(x)
        compute!(kernel, x[i], x[i], target, (i, 1))
    end
end