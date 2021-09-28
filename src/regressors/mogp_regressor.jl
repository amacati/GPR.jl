struct MOGaussianProcessRegressor

    regressors::Vector{GaussianProcessRegressor}
    X::Matrix{Float64}
    _X::Vector{SVector{S, Float64}} where S
    Y::Matrix{Float64}

    function MOGaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel; noisevariance::Float64 = 0.)
        _X = [SVector{size(X,1)}(col) for col in eachcol(X)]
        regressors = [GaussianProcessRegressor(X, _X, Y[i,:], kernel; noisevariance = noisevariance) for i in 1:size(Y,1)]
        new(regressors, X, _X, Y)
    end

    function MOGaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::Vector{<:AbstractKernel}; noisevariance::Vector{Float64} = zeros(size(Y, 1)))
        @assert size(Y,1) == length(kernel) == length(noisevariance) "kernel, noisevariance and Y dimensions have to agree!"
        _X = [SVector{size(X,1), Float64}(col) for col in eachcol(X)]
        regressors = [GaussianProcessRegressor(X, _X, Y[i,:], kernel[i]; noisevariance = noisevariance[i]) for i in 1:size(Y,1)]
        new(regressors, X, _X, Y)
    end
end

function predict(mo_gpr::MOGaussianProcessRegressor, xstart::AbstractArray{Float64}, nsteps::Int)
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
