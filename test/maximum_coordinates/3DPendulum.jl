using Statistics
using GPR
using Plots
using LinearAlgebra

include(joinpath("..", "generatedata.jl"))


storage, mech = simplependulum3D()

data = load3Ddata(storage)
X = data[1][:,1:30:end-1]
Y = data[1][:,2:30:end]

# σ = 1.5, Λ = 1.5 works okayish
kernels = [GeneralGaussianKernel(1.5, ones(size(X,1))*1.5) for _ in 1:size(X,1)]
mo_gpr = MOGaussianProcessRegressor(X, Y, kernels)
optimize!(mo_gpr)
display(mo_gpr.regressors[1].kernel.Λ)


xstart = SVector{size(X,1),Float64}(X[:,1])
μ, σ = predict(mo_gpr, xstart, 999)

xtrue = data[1][:,2:end]
μmat = reshape(reinterpret(Float64, μ), (size(X,1),:))
σmat = reshape(reinterpret(Float64, σ), (size(X,1),:))
error = μmat[1:3,1:999] .- xtrue[1:3,1:999]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

visualize_prediction(mech, [μ])
