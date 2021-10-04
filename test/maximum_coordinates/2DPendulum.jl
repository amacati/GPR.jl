using Statistics
using GPR
using Plots
using LinearAlgebra
using Rotations

include(joinpath("..", "generatedata.jl"))


storage, mech = simplependulum2D()

data = load2Ddata(storage)
X = data[1][:,1:50:end-1]
Y = data[1][:,2:50:end]

kernels = [GeneralGaussianKernel(0.5, ones(size(X,1))*0.5) for _ in 1:size(X,1)]
mo_gpr = MOGaussianProcessRegressor(X, Y, kernels)
optimize!(mo_gpr)

xstart = SVector{size(X,1),Float64}(X[:,1])
μ, σ = predict(mo_gpr, xstart, 999)
xtrue = data[1][:,2:end]
μmat = reshape(reinterpret(Float64, μ), (size(X,1),:))
σmat = reshape(reinterpret(Float64, σ), (size(X,1),:))
error = μmat[1:2,1:999] .- xtrue[1:2,1:999]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

# plot_gp(μmat[1,:], μmat[2,:], sqrt.(σmat[1,:].^2 + σmat[2,:].^2))
# scatter!(reshape(X[1,:],:,1), reshape(X[2,:],:,1), lab="Support points")

function transformTo3D(X)
    ret = [[zeros(length(X[1][1])) for _ in 1:length(X[1])] for _ in 1:length(X)]
    for id in 1:length(X)
        for t in 1:length(X[1])
            state = X[id][t]
            q = UnitQuaternion(RotX(asin(max(min(state[3],1),-1))))
            ret[id][t] = [0, state[1:2]..., q.w, q.x, q.y, q.z]
        end
    end
    return ret
end

visualize_prediction(mech, transformTo3D([μ]))