using LinearAlgebra
using Plots


xtrain = reshape(collect(-5:0.3:5), 1, :)
ytrain = sin.(xtrain)
xtest = reshape(collect(-5:0.01:5), 1, :)

@time gpr = GaussianProcessRegressor(xtrain, ytrain, GaussianKernel(0.8,0.2))
@time mean, sigma = predict(gpr, xtest)

plot_gp(xtest, mean, sigma)
scatter!(reshape(xtrain,:,1), reshape(ytrain,:,1), lab="Support points", legend = :topleft)