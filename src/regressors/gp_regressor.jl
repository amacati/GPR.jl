struct GaussianProcessRegressor

    X::Matrix{Float64}
    _X::Vector{SVector{S, Float64}} where S
    Y::Matrix{Float64}
    _Ymean::Float64
    kernel::AbstractKernel
    noisevariance::Float64
    L::Matrix{Float64}
    α::Matrix{Float64}
    logPY::Float64

    function GaussianProcessRegressor(X::Matrix{Float64}, Y::AbstractArray{Float64}, kernel::AbstractKernel, noisevariance::Float64 = 0.)
        _X = [SVector{size(X,1), Float64}(col) for col in eachcol(X)]
        Y = reshape(Y, 1, :)
        _Ymean = mean(Y)
        Y .-= _Ymean
        L, α = compute_cholesky(_X, Y, kernel, noisevariance)
        logPY = -0.5(Y*α)[1] - sum(log.(diag(L))) - size(Y,2)/2*log(2*pi)
        new(X, _X, Y, _Ymean, kernel, noisevariance, L, α, logPY)
    end

    # Used to share X and _X in MOGaussianProcessRegressors to avoid excessive copying of training points
    function GaussianProcessRegressor(X::Matrix{Float64}, _X::Vector{SVector{S, Float64}}, Y::AbstractArray{Float64}, kernel::AbstractKernel, noisevariance::Float64 = 0.) where S
        Y = reshape(Y, 1, :)
        _Ymean = mean(Y)
        Y .-= _Ymean
        L, α = compute_cholesky(_X, Y, kernel, noisevariance)
        logPY = -0.5(Y*α)[1] - sum(log.(diag(L))) - size(Y,2)/2*log(2*pi)
        new(X, _X, Y, _Ymean, kernel, noisevariance, L, α, logPY)
    end
end

function predict(gpr::GaussianProcessRegressor, xstar::Matrix{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), size(xstar, 2))
    compute_kernelmatrix!(gpr.X, xstar, gpr.kernel, kstar)
    μ = kstar' * gpr.α .+ gpr._Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = Matrix{Float64}(undef, size(xstar,2),1)
    compute_kerneldiagonal!(xstar, gpr.kernel, kdoublestar)
    v = gpr.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::Vector{SVector{S,Float64}}) where S
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), size(xstar, 1))
    compute_kernelmatrix!(gpr._X, xstar, gpr.kernel, kstar)
    μ = kstar' * gpr.α .+ gpr._Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = Matrix{Float64}(undef, size(xstar,1),1)
    compute_kerneldiagonal!(xstar, gpr.kernel, kdoublestar)
    v = gpr.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::Vector{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), 1)
    compute_kernelmatrix!(gpr.X, reshape(xstar,:,1), gpr.kernel, kstar)
    μ = kstar' * gpr.α .+ gpr._Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = Matrix{Float64}(undef, length(xstar), 1)
    compute_kerneldiagonal!(reshape(xstar,:,1), gpr.kernel, kdoublestar)
    v = gpr.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict_full(gpr::GaussianProcessRegressor, xstar::Matrix{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), size(xstar, 2))
    compute_kernelmatrix!(gpr.X, xstar, gpr.kernel, kstar)
    μ = kstar' * gpr.α .+ gpr._Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = Matrix{Float64}(undef, size(xstar,2), size(xstar,2))
    compute_kernelmatrix!(xstar, gpr.kernel, kdoublestar)
    v = gpr.L \ kstar
    σ = kdoublestar - v'*v
    return μ, σ  # σ is the complete covariance matrix
end