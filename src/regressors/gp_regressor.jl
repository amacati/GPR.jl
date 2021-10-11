mutable struct GaussianProcessRegressor

    X::Vector{<:SVector}
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

    function GaussianProcessRegressor(X::Vector{<:SVector}, Y::AbstractArray, kernel::AbstractKernel; noisevariance::Real = 0.)
        Y = reshape(Y, 1, :)
        Ymean = mean(Y)
        Y .-= Ymean

        N = length(X)
        _Kxx = Matrix{Float64}(undef, N, N)
        Kxx = compute_kernelmatrix!(X, kernel, _Kxx)
        chol = cholesky!(Symmetric(Kxx + I*noisevariance, :L))
        α = chol.L'\(chol.L\Y')

        ∇buffer = updatebuffer!(GPRGradientBuffer{N}(), inv(chol), Y)
        parameter_gradient = Vector{Float64}()
        for ∂K∂θi! in get_derivative_handles(kernel)
            ∂K∂θi!(kernel, X, ∇buffer.∂K∂θ)
            push!(parameter_gradient, -0.5*product_trace(∇buffer.ααTK, Symmetric(∇buffer.∂K∂θ, :L), ∇buffer.tracetmp))
        end
        log_marginal_likelihood = -0.5(Y*α)[1] - sum(log.(diag(chol.L))) - N/2*log(2*pi)
        new(X, Y, Ymean, kernel, noisevariance, _Kxx, Kxx, chol, α, ∇buffer, parameter_gradient, log_marginal_likelihood)
    end

end

function updategpr!(gpr, kernel)
    gpr.kernel = kernel
    gpr.Kxx = compute_kernelmatrix!(gpr.X, kernel, gpr._Kxx)
    gpr.chol = cholesky!(Symmetric(gpr.Kxx + I*gpr.noisevariance, :L))
    gpr.α = gpr.chol.L'\(gpr.chol.L\gpr.Y')
    gpr.∇buffer = updatebuffer!(gpr.∇buffer, inv(gpr.chol), gpr.Y)
    parameter_gradient = Vector{Float64}()
    for ∂K∂θi! in get_derivative_handles(gpr.kernel)
        ∂K∂θi!(gpr.kernel, gpr.X, gpr.∇buffer.∂K∂θ)
        push!(parameter_gradient, -0.5*product_trace(gpr.∇buffer.ααTK, Symmetric(gpr.∇buffer.∂K∂θ, :L), gpr.∇buffer.tracetmp))
    end
    gpr.parameter_gradient = parameter_gradient
    gpr.log_marginal_likelihood = -0.5(gpr.Y*gpr.α)[1] - sum(log.(diag(gpr.chol.L))) - size(gpr.Y,2)/2*log(2*pi)
    return nothing
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
    kstar = compute_kernelmatrix(gpr.X, xstar, gpr.kernel)
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