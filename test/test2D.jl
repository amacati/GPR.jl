using LinearAlgebra
using Random
using GPR
using Plots

pyplot()

x1 = collect(repeat(range(-5, length=10, stop=5), outer=10))
x2 = collect(repeat(range(-5, length=10, stop=5), inner=10))
xtrain = vcat(reshape(x1,1,:), reshape(x2,1,:))
ytrain = reshape(sin.(xtrain[1,:]) .* cos.(xtrain[2, :]), 1, :)

# regressor = GaussianProcessRegressor(xtrain, ytrain, GeneralGaussianKernel(2, ones(2)*3))
regressor = GaussianProcessRegressor(xtrain, ytrain, GaussianKernel(2, 0.2))
println("Initial Marginal Likelihood: $(regressor.log_marginal_likelihood)")  # -164.29517869120883
optimize!(regressor)

println("Log Marginal Likelihood: $(regressor.log_marginal_likelihood), σ:$(regressor.kernel.σ) λ:$(regressor.kernel.λ)")
# println("Log Marginal Likelihood: $(regressor.logPY), σ:$(regressor.kernel.σ) Λ:$(regressor.kernel.Λ)")
# Visualization needs callables because of the python interface
f(x,y) = predict(regressor, reshape([x y], :, 1))[1][1][1]  # First tuple + matrix element
ftrue(x,y) = sin(x) * cos(y)
plot_gp((-5,5), (-5,5), f, ftrue)

# 0.008