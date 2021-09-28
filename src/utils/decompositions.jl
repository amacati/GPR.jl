function Lα_decomposition(X::AbstractArray, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Float64)
    K = compute_kernelmatrix(X, kernel)
    for _ in 1:10
        try
            L = cholesky!(Symmetric(K + I*noisevariance, :L)).L
            return L, L'\(L\Y')  # Also precompute α needed for predictions
        catch PosDefException
            @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
            noisevariance += 1e-9
        end
    end
    error("Cholesky decomposition impossible")
end

function Lα_decomposition!(X::AbstractArray, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Float64, K::Matrix)
    compute_kernelmatrix!(X, kernel, K)
    for _ in 1:10
        try
            L = cholesky!(Symmetric(K + I*noisevariance, :L)).L
            return L, L'\(L\Y')  # Also precompute α needed for predictions
        catch PosDefException
            @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
            noisevariance += 1e-9
        end
    end
    error("Cholesky decomposition impossible")
end

