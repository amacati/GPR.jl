mutable struct GaussianProcessRegressor

    X::AbstractMatrix
    _X::Vector{SVector{S, T}} where {S,T}
    Y::AbstractMatrix
    _Ymean::Real
    kernel::AbstractKernel
    noisevariance::Real
    L::AbstractMatrix
    α::AbstractMatrix
    logPY::Real

    function GaussianProcessRegressor(X::AbstractMatrix, Y::AbstractArray, kernel::AbstractKernel; noisevariance::Real = 0.)
        _X = [SVector{size(X,1), eltype(X)}(col) for col in eachcol(X)]
        Y = reshape(Y, 1, :)
        _Ymean = mean(Y)
        Y .-= _Ymean
        L, α = Lα_decomposition(_X, Y, kernel, noisevariance)
        logPY = -0.5(Y*α)[1] - sum(log.(diag(L))) - size(Y,2)/2*log(2*pi)
        new(X, _X, Y, _Ymean, kernel, noisevariance, L, α, logPY)
    end

    # Used to share X and _X in MOGaussianProcessRegressors to avoid copying of training points
    function GaussianProcessRegressor(X::AbstractMatrix, _X::Vector{SVector{S, T}}, Y::AbstractArray, kernel::AbstractKernel; noisevariance::Real = 0.) where {S, T}
        Y = reshape(Y, 1, :)
        _Ymean = mean(Y)
        Y .-= _Ymean
        L, α = Lα_decomposition(_X, Y, kernel, noisevariance)
        logPY = -0.5(Y*α)[1] - sum(log.(diag(L))) - size(Y,2)/2*log(2*pi)
        new(X, _X, Y, _Ymean, kernel, noisevariance, L, α, logPY)
    end
end

function predict(gpr::GaussianProcessRegressor, xstar::AbstractMatrix)
    kstar = compute_kernelmatrix(gpr.X, xstar, gpr.kernel)
    μ = kstar' * gpr.α .+ gpr._Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = compute_kerneldiagonal(xstar, gpr.kernel)
    v = gpr.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::AbstractVector)
    return predict(gpr, reshape(xstar, :, 1))
end

function predict(gpr::GaussianProcessRegressor, xstar::Vector{SVector{S,T}}) where {S,T}
    kstar = compute_kernelmatrix(gpr._X, xstar, gpr.kernel)
    μ = kstar' * gpr.α .+ gpr._Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = compute_kerneldiagonal(xstar, gpr.kernel)
    v = gpr.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::SVector{S, T}) where {S,T}
    return predict(gpr, [xstar,])
end

function predict_full(gpr::GaussianProcessRegressor, xstar::AbstractMatrix)
    kstar = compute_kernelmatrix(gpr.X, xstar, gpr.kernel)
    μ = kstar' * gpr.α .+ gpr._Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = compute_kernelmatrix(xstar, gpr.kernel)
    v = gpr.L \ kstar
    σ = kdoublestar - v'*v
    return μ, σ  # σ is the complete covariance matrix
end