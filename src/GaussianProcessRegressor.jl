module GPR

include("Kernel.jl")
include("utils/kernelmatrix.jl")
include("utils/cholesky.jl")

export GaussianProcessRegressor
export GaussianKernel
export predict
export predict_full


struct GaussianProcessRegressor

    X::Matrix{Float64}
    Y::Matrix{Float64}
    kernel::AbstractKernel
    noisevariance::Number
    L::Matrix{Float64}
    α::Matrix{Float64}

    GaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel) = new(X, Y, kernel, 0, compute_cholesky(X, Y, kernel, 0)...)
    GaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::AbstractKernel, noisevariance) = new(X, Y, kernel, noisevariance, compute_cholesky(X, Y, kernel, noisevariance)...)

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