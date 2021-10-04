struct MOGaussianProcessRegressor

    regressors::Vector{GaussianProcessRegressor}
    X::AbstractMatrix
    _X::Vector{SVector{S, T}} where {S,T}
    Y::AbstractMatrix
    regressorcount::Int

    function MOGaussianProcessRegressor(X::AbstractMatrix, Y::AbstractMatrix, kernel::AbstractKernel; noisevariance::Real = 0.)
        _X = [SVector{size(X,1)}(col) for col in eachcol(X)]
        regressors = [GaussianProcessRegressor(X, _X, Y[i,:], kernel; noisevariance = noisevariance) for i in 1:size(Y,1)]
        new(regressors, X, _X, Y, length(regressors))
    end

    function MOGaussianProcessRegressor(X::AbstractMatrix, Y::AbstractMatrix, kernel::Vector{<:AbstractKernel}; noisevariance::Vector{<:Real} = zeros(size(Y, 1)))
        @assert size(Y,1) == length(kernel) == length(noisevariance) "kernel, noisevariance and Y dimensions have to agree!"
        _X = [SVector{size(X,1), Float64}(col) for col in eachcol(X)]
        regressors = [GaussianProcessRegressor(X, _X, Y[i,:], kernel[i]; noisevariance = noisevariance[i]) for i in 1:size(Y,1)]
        new(regressors, X, _X, Y, length(regressors))
    end
end

Base.iterate(mogpr::MOGaussianProcessRegressor, state=1) = state > mogpr.regressorcount ? nothing : (mogpr.regressors[state], state+1)

function predict(mo_gpr::MOGaussianProcessRegressor, xstart::AbstractArray, nsteps::Int)
    N = length(mo_gpr.regressors)
    μ = Vector{SVector{N,Float64}}(undef, nsteps)
    σ = Vector{SVector{N,Float64}}(undef, nsteps)
    μtemp = Vector{Float64}(undef, N)
    σtemp = Vector{Float64}(undef, N)
    for j in 1:N
        # map extracts the float values from tuple of 1x1 matrices of predict
        μtemp[j], σtemp[j] = map(a-> a[1][1], predict(mo_gpr.regressors[j], xstart))
    end
    μ[1] = SVector{N,Float64}(μtemp)
    σ[1] = SVector{N,Float64}(σtemp)
    for i in 2:nsteps
        for j in 1:N
            μtemp[j], σtemp[j] = map(a -> a[1][1], predict(mo_gpr.regressors[j], μ[i-1]))
        end
        μ[i] = SVector{N,Float64}(μtemp)
        σ[i] = SVector{N,Float64}(σtemp)
    end
    return μ, σ
end
