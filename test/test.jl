using LinearAlgebra

include("../src/GaussianProcessRegressor.jl")
include("../src/visualization/visualization.jl")

xtrain = reshape(collect(0:0.1:6), 1, :)
ytrain = sin.(xtrain)

xtest = reshape(collect(-1:0.1:7), 1, :)

display(size(xtest))

gpr = GPR.GaussianProcessRegressor(xtrain, ytrain, GPR.GaussianKernel(0.5,1.0), 0.5)

mean, sigma = GPR.predict(gpr, xtest)

plot_gp(reshape(xtest,:,1), mean, diag(sigma))
scatter!(reshape(xtrain,:,1), reshape(ytrain,:,1), lab="Support points", legend = :topleft)