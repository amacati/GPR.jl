using LinearAlgebra
include("../Kernel.jl")


function compute_cholesky(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Number)::Tuple{Matrix{Float64}, Matrix{Float64}}
    k = Matrix{Float64}(undef, size(X, 2), size(X, 2))
    compute_kernelmatrix!(X, kernel, k)
    L = cholesky!(Symmetric(k + I*noisevariance, :L)).L
    return L, L'\(L\Y')  # Also precompute α needed for predictions
end