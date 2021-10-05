mutable struct GaussianProcessRegressor

    X::AbstractMatrix
    Xstatic::Vector{SVector{S, T}} where {S,T}
    Y::AbstractMatrix
    Ymean::Real
    kernel::AbstractKernel
    noisevariance::Real
    _Kxx::AbstractMatrix  # Dense matrix with lower triangular calculated
    Kxx::AbstractMatrix  # Symmetric view of _kxx for computations
    chol::Cholesky  # Cholesky decomposition
    α::AbstractMatrix
    ∇buffer::GPRGradientBuffer
    parameter_gradient::AbstractVector
    log_marginal_likelihood::Real

    function GaussianProcessRegressor(X::AbstractMatrix, Y::AbstractArray, kernel::AbstractKernel; noisevariance::Real = 0.)
        N1, N2, T = size(X)..., eltype(X)
        Xstatic = [SVector{N1, T}(col) for col in eachcol(X)]
        Y = reshape(Y, 1, :)
        Ymean = mean(Y)
        Y .-= Ymean

        _Kxx = Matrix{Float64}(undef, N2, N2)
        Kxx = compute_kernelmatrix!(Xstatic, kernel, _Kxx)
        chol = cholesky!(Symmetric(Kxx + I*noisevariance, :L))
        α = chol.L'\(chol.L\Y')

        ∇buffer = updatebuffer!(GPRGradientBuffer{N2}(), inv(chol), Y)
        parameter_gradient = Vector{Float64}()
        for ∂K∂θi! in get_derivative_handles(kernel)
            ∂K∂θi!(kernel, Xstatic, ∇buffer.∂K∂θ)
            push!(parameter_gradient, -0.5*product_trace(∇buffer.ααTK, Symmetric(∇buffer.∂K∂θ, :L), ∇buffer.tracetmp))
        end
        log_marginal_likelihood = -0.5(Y*α)[1] - sum(log.(diag(chol.L))) - N2/2*log(2*pi)
        new(X, Xstatic, Y, Ymean, kernel, noisevariance, _Kxx, Kxx, chol, α, ∇buffer, parameter_gradient, log_marginal_likelihood)
    end

    # Used to share X and _X in MOGaussianProcessRegressors to avoid copying of training points
    function GaussianProcessRegressor(X::AbstractMatrix, _X::Vector{SVector{S, T}}, Y::AbstractArray, kernel::AbstractKernel; noisevariance::Real = 0.) where {S, T}
        Y = reshape(Y, 1, :)
        Ymean = mean(Y)
        Y .-= Ymean
        L, α = Lα_decomposition(Xstatic, Y, kernel, noisevariance)
        log_marginal_likelihood = -0.5(Y*α)[1] - sum(log.(diag(L))) - size(Y,2)/2*log(2*pi)
        new(X, Xstatic, Y, Ymean, kernel, noisevariance, L, α, log_marginal_likelihood)
    end
end

function updategpr!(gpr, kernel)
    gpr.kernel = kernel
    gpr.Kxx = compute_kernelmatrix!(gpr.Xstatic, kernel, gpr._Kxx)
    gpr.chol = cholesky!(Symmetric(gpr.Kxx + I*gpr.noisevariance, :L))
    gpr.α = gpr.chol.L'\(gpr.chol.L\gpr.Y')
    gpr.∇buffer = updatebuffer!(gpr.∇buffer, inv(gpr.chol), gpr.Y)
    parameter_gradient = Vector{Float64}()
    for ∂K∂θi! in get_derivative_handles(gpr.kernel)
        ∂K∂θi!(gpr.kernel, gpr.Xstatic, gpr.∇buffer.∂K∂θ)
        push!(parameter_gradient, -0.5*product_trace(gpr.∇buffer.ααTK, Symmetric(gpr.∇buffer.∂K∂θ, :L), gpr.∇buffer.tracetmp))
    end
    gpr.parameter_gradient = parameter_gradient
    gpr.log_marginal_likelihood = -0.5(gpr.Y*gpr.α)[1] - sum(log.(diag(gpr.chol.L))) - size(gpr.Y,2)/2*log(2*pi)
end

function predict(gpr::GaussianProcessRegressor, xstar::AbstractMatrix)
    kstar = compute_kernelmatrix(gpr.X, xstar, gpr.kernel)
    μ = kstar' * gpr.α .+ gpr.Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = compute_kerneldiagonal(xstar, gpr.kernel)
    v = gpr.chol.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::AbstractVector)
    return predict(gpr, reshape(xstar, :, 1))
end

function predict(gpr::GaussianProcessRegressor, xstar::Vector{SVector{S,T}}) where {S,T}
    kstar = compute_kernelmatrix(gpr.Xstatic, xstar, gpr.kernel)
    μ = kstar' * gpr.α .+ gpr.Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = compute_kerneldiagonal(xstar, gpr.kernel)
    v = gpr.chol.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::SVector{S, T}) where {S,T}
    return predict(gpr, [xstar,])
end

function predict_full(gpr::GaussianProcessRegressor, xstar::AbstractMatrix)
    kstar = compute_kernelmatrix(gpr.X, xstar, gpr.kernel)
    μ = kstar' * gpr.α .+ gpr.Ymean  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = compute_kernelmatrix(xstar, gpr.kernel)
    v = gpr.chol.L \ kstar
    σ = kdoublestar - v'*v
    return μ, σ  # σ is the complete covariance matrix
end