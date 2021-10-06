mutable struct GaussianProcessQuaternionRegressor

    X::Vector{UnitQuaternion}
    Y::Vector{UnitQuaternion}
    Ymean::UnitQuaternion
    kernel::AbstractKernel
    noisevariance::Real
    _Kxx::AbstractMatrix  # Dense matrix with lower triangular calculated
    Kxx::AbstractMatrix  # Symmetric view of _kxx for computations
    chol::Cholesky  # Cholesky decomposition
    Kinv::AbstractMatrix
    parameter_gradient::AbstractVector
    log_marginal_likelihood::Real

    function GaussianProcessQuaternionRegressor(X::Vector{<:UnitQuaternion}, Y::Vector{<:UnitQuaternion}, kernel::AbstractKernel; noisevariance::Real = 0.)
        N = length(X)
        Ymean = quaternion_average(Y)

        _Kxx = Matrix{Float64}(undef, N, N)
        Kxx = compute_kernelmatrix!(X, kernel, _Kxx)
        chol = cholesky!(Symmetric(Kxx + I*noisevariance, :L))
        Kinv = inv(chol)
        parameter_gradient = zeros(2)
        new(X, Y, Ymean, kernel, noisevariance, _Kxx, Kxx, chol, Kinv, parameter_gradient, 0)
    end

end

function predict(gpr::GaussianProcessQuaternionRegressor, xstar::UnitQuaternion)
    kstar = compute_kernelmatrix(gpr.X, [xstar], gpr.kernel)

    weights = max.(kstar' * gpr.Kinv,0)  # Numerical errors can lead to weights < 0 which should not be possible
    μ = quaternion_average(gpr.Y, vec(weights), gpr.Ymean)
    return μ
end
