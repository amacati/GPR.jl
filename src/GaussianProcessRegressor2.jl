module GPR

include("Kernel.jl")

export GaussianProcessRegressor
export GaussianKernel
export predict


function compute_kinv(X::AbstractMatrix, kernel::GaussianKernel, noisevariance::Number)
    nelements = size(X, 2)
    k = zeros(nelements, nelements)
    for i = 1:nelements, j = 1:nelements
        k[i, j] = compute(kernel, X[:,i], X[:, j])
    end
    return inv(k + I*noisevariance)
end

struct GaussianProcessRegressor

    X::AbstractArray
    Y::AbstractArray
    kernel
    noisevariance::Number
    kinv::AbstractMatrix

    GaussianProcessRegressor(X::AbstractArray, Y::AbstractArray, kernel) = new(X, Y, kernel, 0, compute_kinv(X, kernel, 0))
    GaussianProcessRegressor(X::AbstractArray, Y::AbstractArray, kernel, noisevariance) = new(X, Y, kernel, noisevariance, compute_kinv(X, kernel, noisevariance))

end

function predict(gpr::GaussianProcessRegressor, x::AbstractArray)
    kstar = zeros(size(gpr.X, 2), size(x, 2))
    for i in 1:size(gpr.X, 2), j = 1:size(x, 2)
        kstar[i, j] = compute(gpr.kernel, gpr.X[:,i], x[:,j])
    end
    μ = transpose(kstar)*gpr.kinv*transpose(gpr.Y)

    kdoublestar = zeros(size(x,2), size(x,2))
    for i in 1:size(x, 2), j = 1:size(x, 2)
        kdoublestar[i, j] = compute(gpr.kernel, x[:,i], x[:,j])
    end
    σ = kdoublestar - transpose(kstar)*gpr.kinv*kstar
    return μ, σ
end


end