using Statistics
using GPR
using Plots
using LinearAlgebra

include(joinpath("..", "generatedata.jl"))

xtrain = (rand(2,100) .- 0.5) .* 12
ytrain = sin.(xtrain) .* cos.(xtrain)
#=
mo_gpr = MOGaussianProcessRegressor(xtrain, ytrain, GaussianKernel(0.5,1.0))

xstart = SVector{size(xtrain,1), Float64}(xtrain[:,1])
μ, σ = predict(mo_gpr, xstart, 20)

display(μ)
=#
storage, mech = simplependulum2D()
ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")

data = load2Ddata(storage)
display(data[1])