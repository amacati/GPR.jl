function Lα_decomposition(X::AbstractMatrix, Y::AbstractMatrix, kernel::AbstractKernel, noisevariance::Real)
    Ksym = compute_kernelmatrix(X, kernel)
    for _ in 1:10
        try
            L = cholesky!(Symmetric(Ksym + I*noisevariance, :L)).L
            return L, L'\(L\Y')  # Also precompute α needed for predictions
        catch PosDefException
            @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
            noisevariance += 1e-9
        end
    end
    error("Cholesky decomposition impossible")
end

function Lα_decomposition!(X::AbstractMatrix, Y::AbstractMatrix, kernel::AbstractKernel, noisevariance::Real, K::AbstractMatrix)
    Ksym = compute_kernelmatrix!(X, kernel, K)
    for _ in 1:10
        try
            L = cholesky!(Symmetric(Ksym + I*noisevariance, :L)).L
            return L, L'\(L\Y')  # Also precompute α needed for predictions
        catch PosDefException
            @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
            noisevariance += 1e-9
        end
    end
    error("Cholesky decomposition impossible")
end

function Lα_decomposition(X::Vector{SVector{S,T}}, Y::AbstractMatrix, kernel::AbstractKernel, noisevariance::Real) where {S,T}
    Ksym = compute_kernelmatrix(X, kernel)
    for _ in 1:10
        try
            L = cholesky!(Symmetric(Ksym + I*noisevariance, :L)).L
            return L, L'\(L\Y')  # Also precompute α needed for predictions
        catch PosDefException
            @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
            noisevariance += 1e-9
        end
    end
    error("Cholesky decomposition impossible")
end

function Lα_decomposition!(X::Vector{SVector{S,T}}, Y::AbstractMatrix, kernel::AbstractKernel, noisevariance::Real, K::AbstractMatrix) where {S,T}
    Ksym = compute_kernelmatrix!(X, kernel, K)
    for _ in 1:10
        try
            L = cholesky!(Symmetric(Ksym + I*noisevariance, :L)).L
            return L, L'\(L\Y')  # Also precompute α needed for predictions
        catch PosDefException
            @warn "K not PSD, cholesky decomposition failed. Adding residuals to diagonal for stabilization." maxlog=1
            noisevariance += 1e-9
        end
    end
    error("Cholesky decomposition impossible")
end
