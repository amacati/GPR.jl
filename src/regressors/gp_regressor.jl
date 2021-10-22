mutable struct GaussianProcessRegressor

    X::Vector{<:SVector}
    Y::AbstractMatrix
    kernel::AbstractKernel
    noisevariance::Real
    parametergradient::AbstractVector
    log_marginal_likelihood::Real
    μ::Function

    _μY::Real
    _Kxxdense::AbstractMatrix  # Dense matrix with lower triangular calculated
    _Kxx::Symmetric  # Symmetric view of _Kxxdense for computations
    _Chol::Cholesky  # Cholesky decomposition
    _α::AbstractMatrix
    _∇buffer::GPRGradientBuffer

    function GaussianProcessRegressor(X::Vector{<:SVector}, Y::AbstractArray, kernel::AbstractKernel; noisevariance::Real = 0., μ::Function = _ -> 0)
        Y = reshape(Y, 1, :)
        Y .-= reshape([μ(state) for state in X], 1, :)  # Substract mean function from Y before computing any statistics on it
        _μY = mean(Y)
        Y .-= _μY

        N = length(X)
        _Kxxdense = Matrix{Float64}(undef, N, N)
        _Kxx = compute_kernelmatrix!(X, kernel, _Kxxdense)
        _Chol = cholesky!(Symmetric(_Kxx + I*noisevariance, :L))
        _α = _Chol.L'\(_Chol.L\Y')

        _∇buffer = updatebuffer!(GPRGradientBuffer{N}(), inv(_Chol), Y)
        parametergradient = Vector{Float64}()
        for ∂K∂θi! in get_derivative_handles(kernel)
            ∂K∂θi!(kernel, X, _∇buffer.∂K∂θ)
            push!(parametergradient, -0.5*product_trace(_∇buffer.ααTK, Symmetric(_∇buffer.∂K∂θ, :L), _∇buffer.tracetmp))
        end
        log_marginal_likelihood = -0.5(Y*_α)[1] - sum(log.(diag(_Chol.L))) - N/2*log(2*pi)
        new(X, Y, kernel, noisevariance, parametergradient, log_marginal_likelihood, μ, _μY,_Kxxdense, _Kxx, _Chol, _α, _∇buffer)
    end
end

function updategpr!(gpr, kernel)
    gpr.kernel = kernel
    gpr._Kxx = compute_kernelmatrix!(gpr.X, kernel, gpr._Kxxdense)
    gpr._Chol = cholesky!(Symmetric(gpr._Kxx + I*gpr.noisevariance, :L))
    gpr._α = gpr._Chol.L'\(gpr._Chol.L\gpr.Y')
    gpr._∇buffer = updatebuffer!(gpr._∇buffer, inv(gpr._Chol), gpr.Y)
    parametergradient = Vector{Float64}()
    for ∂K∂θi! in get_derivative_handles(gpr.kernel)
        ∂K∂θi!(gpr.kernel, gpr.X, gpr._∇buffer.∂K∂θ)
        push!(parametergradient, -0.5*product_trace(gpr._∇buffer.ααTK, Symmetric(gpr._∇buffer.∂K∂θ, :L), gpr._∇buffer.tracetmp))
    end
    gpr.parametergradient = parametergradient
    gpr.log_marginal_likelihood = -0.5(gpr.Y*gpr._α)[1] - sum(log.(diag(gpr._Chol.L))) - size(gpr.Y,2)/2*log(2*pi)
    return nothing
end

function predict(gpr::GaussianProcessRegressor, xstar::Vector{SVector{S,T}}) where {S,T}
    kstar = compute_kernelmatrix(gpr.X, xstar, gpr.kernel)
    μ = kstar' * gpr._α .+ gpr._μY .+ [gpr.μ(x) for x in xstar] # Add mean of Y to retransform into non-zero average output space
    kdoublestar = compute_kerneldiagonal(xstar, gpr.kernel)
    v = gpr._Chol.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::SVector{S, T}) where {S,T}
    return predict(gpr, [xstar,])
end

function predict_full(gpr::GaussianProcessRegressor, xstar::AbstractMatrix)
    kstar = compute_kernelmatrix(gpr.X, xstar, gpr.kernel)
    μ = kstar' * gpr._α .+ gpr._μY  # Add mean of Y to retransform into non-zero average output space

    kdoublestar = compute_kernelmatrix(xstar, gpr.kernel)
    v = gpr._Chol.L \ kstar
    σ = kdoublestar - v'*v
    return μ, σ  # σ is the complete covariance matrix
end