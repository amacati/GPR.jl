module GPR

include("kernels/AbstractKernel.jl")
include("kernels/GaussianKernel.jl")
include("utils/kernelmatrix.jl")
include("utils/cholesky.jl")
include("visualization/visualization.jl")

export GaussianProcessRegressor
export GaussianKernel
export GeneralGaussianKernel
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

    function GaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance::Float64 = 0.)
        L, α = compute_cholesky(X, Y, kernel, noisevariance)
        new(X, Y, kernel, noisevariance, L, α)
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