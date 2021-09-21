using LinearAlgebra
using Random
using GPR
using Plots

pyplot()

xtrain = (rand(2,100) .- 0.5) .* 12
ytrain = reshape(sin.(xtrain[1,:]) .* cos.(xtrain[2, :]), 1, :)

regressor = GaussianProcessRegressor(xtrain, ytrain, GaussianKernel(0.5,1.0), 0.5)

# Visualization needs callables because of the python interface
f(x,y) = predict(regressor, reshape([x y], :, 1))[1][1][1]  # First tuple + matrix element
ftrue(x,y) = sin(x) * cos(y)
plot_gp((-5,5), (-5,5), f, ftrue)
