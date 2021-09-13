using LinearAlgebra
using Random

include("../src/GaussianProcessRegressor.jl")
include("../src/visualization/visualization.jl")

xtrain = (rand(2,100) .- 0.5) .* 12
ytrain = reshape(sin.(xtrain[1,:]) .* cos.(xtrain[2, :]), 1, :)

gpr = GPR.GaussianProcessRegressor(xtrain, ytrain, GPR.GaussianKernel(0.5,1.0), 0.5)

ftrue(x,y) = sin(x) * cos(y)
plot_gp((-5,5), (-5,5), gpr, ftrue)
