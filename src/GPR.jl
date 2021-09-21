module GPR

include("kernels/AbstractKernel.jl")
include("kernels/GaussianKernel.jl")
include("kernels/MaternKernel.jl")
include("utils/kernelmatrix.jl")
include("utils/cholesky.jl")
include("visualization/visualization.jl")

export GaussianProcessRegressor
export GaussianKernel
export GeneralGaussianKernel
export MaternKernel
export predict
export predict_full
export plot_gp


struct GaussianProcessRegressor

    X::Matrix{Float64}
    Y::Matrix{Float64}
    kernel::AbstractKernel
    noisevariance::Float64
    L::Matrix{Float64}
    α::Matrix{Float64}
    logPY::Float64

    function GaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Float64 = 0.)
        L, α = compute_cholesky(X, Y, kernel, noisevariance)
        logPY = -0.5(Y*α)[1] - sum(log.(diag(L))) - size(Y,2)/2*log(2*pi)
        new(X, Y, kernel, noisevariance, L, α, logPY)
    end

    function GaussianProcessRegressor(X::Matrix{Float64}, Y::Vector{Float64}, kernel::AbstractKernel, noisevariance::Float64 = 0.)
        Y = reshape(Y, 1, :)
        L, α = compute_cholesky(X, Y, kernel, noisevariance)
        logPY = -0.5(Y*α)[1] - sum(log.(diag(L))) - size(Y,2)/2*log(2*pi)
        new(X, Y, kernel, noisevariance, L, α, logPY)
    end
end

function predict(gpr::GaussianProcessRegressor, xstar::Matrix{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), size(xstar, 2))
    compute_kernelmatrix!(gpr.X, xstar, gpr.kernel, kstar)
    μ = kstar' * gpr.α

    kdoublestar = Matrix{Float64}(undef, size(xstar,2),1)
    compute_kerneldiagonal!(xstar, gpr.kernel, kdoublestar)
    v = gpr.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gpr::GaussianProcessRegressor, xstar::Vector{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), 1)
    compute_kernelmatrix!(gpr.X, reshape(xstar,:,1), gpr.kernel, kstar)
    μ = kstar' * gpr.α

    kdoublestar = Matrix{Float64}(undef, length(xstar), 1)
    compute_kerneldiagonal!(reshape(xstar,:,1), gpr.kernel, kdoublestar)
    v = gpr.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gprs::AbstractArray{GaussianProcessRegressor}, xstart::AbstractArray{Float64},
                 nsteps::Int, Ymean::AbstractArray{Float64})
    statesize = length(gprs)
    μ = Matrix{Float64}(undef, statesize, nsteps)
    σ = Matrix{Float64}(undef, statesize, nsteps)
    for j in 1:statesize
        # map extracts the float values from tuple of 1x1 matrices of predict
        μ[j, 1], σ[j, 1] = map(a-> a[1][1], predict(gprs[j], reshape(xstart,:,1)))
    end
    for i in 2:nsteps, j in 1:statesize
        μ[j, i], σ[j, i] = map(a -> a[1][1], predict(gprs[j], μ[:,i-1] + Ymean))
    end
    return μ, σ
end

function predict_full(gpr::GaussianProcessRegressor, xstar::Matrix{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), size(xstar, 2))
    compute_kernelmatrix!(gpr.X, xstar, gpr.kernel, kstar)
    μ = kstar' * gpr.α

    kdoublestar = Matrix{Float64}(undef, size(xstar,2), size(xstar,2))
    compute_kernelmatrix!(xstar, gpr.kernel, kdoublestar)
    v = gpr.L \ kstar
    σ = kdoublestar - v'*v
    return μ, σ  # σ is the complete covariance matrix
end

end