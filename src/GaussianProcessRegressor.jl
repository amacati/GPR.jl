module GPR

include("Kernel.jl")

export GaussianProcessRegressor
export GaussianKernel
export predict
export predict_full


function compute_cholesky(X::Matrix{Float64}, Y::Matrix{Float64}, kernel::GaussianKernel, noisevariance::Number)
    nelements = size(X, 2)
    k = Matrix{Float64}(undef, nelements, nelements)
    for i = 1:nelements, j = 1:i
        compute!(kernel, X[:,i], X[:,j], k, (i, j))
    end
    C = cholesky!(Symmetric(k + I*noisevariance, :L))
    α = C.L'\(C.L\Y')
    return C, α
end

struct GaussianProcessRegressor

    X::Matrix{Float64}
    Y::Matrix{Float64}
    kernel
    noisevariance::Number
    C::Cholesky
    α::Matrix{Float64}

    GaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel) = new(X, Y, kernel, 0, compute_cholesky(X, Y, kernel, 0)...)
    GaussianProcessRegressor(X::Matrix{Float64}, Y::Matrix{Float64}, kernel, noisevariance) = new(X, Y, kernel, noisevariance, compute_cholesky(X, Y, kernel, noisevariance)...)

end

function predict(gpr::GaussianProcessRegressor, x::Matrix{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), size(x, 2))
    for i in 1:size(gpr.X, 2), j = 1:size(x, 2)
        compute!(gpr.kernel, gpr.X[:,i], x[:,j], kstar, (i,j))
    end
    μ = kstar' * gpr.α

    kdoublestar = Matrix{Float64}(undef, size(x,2),1)
    for i in 1:size(x, 2)
        compute!(gpr.kernel, x[:,i], x[:,i], kdoublestar, (i,1))
    end
    v = gpr.C.L \ kstar
    σ = kdoublestar .- diag(v'*v)
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict_full(gpr::GaussianProcessRegressor, x::Matrix{Float64})
    kstar = Matrix{Float64}(undef, size(gpr.X, 2), size(x, 2))
    for i in 1:size(gpr.X, 2), j = 1:size(x, 2)
        compute!(gpr.kernel, gpr.X[:,i], x[:,j], kstar, (i,j))
    end
    μ = kstar' * gpr.α

    kdoublestar = Matrix{Float64}(undef, size(x,2), size(x,2))
    for i in 1:size(x, 2), j = 1:i
        compute!(gpr.kernel, x[:,i], x[:,j], kdoublestar, (i,j))
    end
    kdoublestar = Symmetric(kdoublestar, :L)
    v = gpr.C.L \ kstar
    σ = kdoublestar - v'*v
    return μ, σ  # σ is the complete covariance matrix
end


end