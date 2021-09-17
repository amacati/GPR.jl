using LinearAlgebra
include("../kernels/AbstractKernel.jl")

function compute_cholesky(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Number)::Tuple{Matrix{Float64}, Matrix{Float64}}
    k = Matrix{Float64}(undef, size(X, 2), size(X, 2))
    compute_kernelmatrix!(X, kernel, k)
    try
        L = cholesky!(Symmetric(k + I*noisevariance, :L)).L
        return L, L'\(L\Y')  # Also precompute α needed for predictions
    catch PosDefException
        @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization."
        L = cholesky!(Symmetric(k + I*noisevariance + I*1e-5, :L)).L
        return L, L'\(L\Y')  # Also precompute α needed for predictions
    end
end