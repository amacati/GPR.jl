function compute_cholesky(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Float64)
    k = Matrix{Float64}(undef, size(X, 2), size(X, 2))
    compute_kernelmatrix!(X, kernel, k)
    try
        L = cholesky!(Symmetric(k + I*noisevariance, :L)).L
        return L, L'\(L\Y')  # Also precompute α needed for predictions
    catch PosDefException
        @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
        L = cholesky!(Symmetric(k + I*noisevariance + I*1e-9, :L)).L
        return L, L'\(L\Y')  # Also precompute α needed for predictions
    end
end

function compute_cholesky(X::Vector{SVector{S,Float64}}, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Float64) where {S}
    k = Matrix{Float64}(undef, size(X,1), size(X,1))
    compute_kernelmatrix!(X, kernel, k)
    try
        L = cholesky!(Symmetric(k + I*noisevariance, :L)).L
        return L, L'\(L\Y')  # Also precompute α needed for predictions
    catch PosDefException
        @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
        L = cholesky!(Symmetric(k + I*noisevariance + I*1e-9, :L)).L
        return L, L'\(L\Y')  # Also precompute α needed for predictions
    end
end
