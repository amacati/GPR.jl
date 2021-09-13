using LinearAlgebra

include("../src/GaussianProcessRegressor.jl")
include("../src/visualization/visualization.jl")

xtrain = (rand(1,20) .- 0.5) .* 10
ytrain = sin.(xtrain)
xtest = reshape(collect(-5:0.1:5), 1, :)

gpr = GPR.GaussianProcessRegressor(xtrain, ytrain, GPR.GaussianKernel(1.0,0.6))
mean, sigma = GPR.predict(gpr, xtest)

plot_gp(xtest, mean, diag(sigma))
scatter!(reshape(xtrain,:,1), reshape(ytrain,:,1), lab="Support points", legend = :topleft)