using LinearAlgebra
using Plots
using GPR


xtrain = reshape(collect(-5:0.3:5), 1, :)
ytrain = sin.(xtrain)
xtest = reshape(collect(-5:0.01:5), 1, :)


regressor = GaussianProcessRegressor(xtrain, ytrain, GaussianKernel(0.5, 0.5))
optimize!(regressor)
μ, σ = predict(regressor, xtest)

println("Maginal log-likelihood: $(regressor.logPY)")
println("Kernel params: σ: $(regressor.kernel.σ), λ: $(regressor.kernel.λ)")

plot_gp(xtest, μ, σ)
scatter!(reshape(xtrain,:,1), reshape(ytrain,:,1), lab="Support points", legend = :topleft)
