using Statistics
using GPR
using Plots
using LinearAlgebra

include(joinpath("..", "generatedata.jl"))


storage, mech = simplependulum()
data = load2Ddata(storage)
X = data[1][:,1:50:end-1]
Y = data[1][:,2:50:end]

kernels = Vector{GeneralGaussianKernel}(undef, 6)
for i in 1:6
    kernels[i] = GeneralGaussianKernel(0.5, ones(6)*0.5)
end
mo_gpr = MOGaussianProcessRegressor(X, Y, kernels)
optimize!(mo_gpr)

xstart = SVector{size(X,1),Float64}(X[:,1])
μstatic, σstatic = predict(mo_gpr, xstart, 200)
xtrue = data[1][:,2:end]

μ = zeros(length(μstatic[1]), length(μstatic))
σ = zeros(length(σstatic[1]), length(σstatic))
for i in 1:length(μstatic)
    μ[:,i] = μstatic[i]
    σ[:,i] = σstatic[i]
end

error = μ[1:2,1:200] .- xtrue[1:2,1:200]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

plot_gp(μ[1,:], μ[2,:], sqrt.(σ[1,:].^2 + σ[2,:].^2))
scatter!(reshape(X[1,:],:,1), reshape(X[2,:],:,1), lab="Support points")

# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
