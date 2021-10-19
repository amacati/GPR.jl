using GPR
using LinearAlgebra
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


storage, mechanism, initialstates = doublependulum3D()
data = loaddata(storage)
resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position
steps = 11
X = data[2:steps:end-1]
Yv11 = [sample[8] for sample in data[3:steps:end]]
Yv12 = [sample[9] for sample in data[3:steps:end]]
Yv13 = [sample[10] for sample in data[3:steps:end]]
Yω11 = [sample[11] for sample in data[3:steps:end]]
Yω12 = [sample[12] for sample in data[3:steps:end]]
Yω13 = [sample[13] for sample in data[3:steps:end]]
Yv21 = [sample[8] for sample in data[3:steps:end]]
Yv22 = [sample[9] for sample in data[3:steps:end]]
Yv23 = [sample[10] for sample in data[3:steps:end]]
Yω21 = [sample[11] for sample in data[3:steps:end]]
Yω22 = [sample[12] for sample in data[3:steps:end]]
Yω23 = [sample[13] for sample in data[3:steps:end]]

kernel = GeneralGaussianKernel(0.5, ones(26)*0.5)
gprv11 = GaussianProcessRegressor(X, Yv11, copy(kernel))
gprv12 = GaussianProcessRegressor(X, Yv12, copy(kernel))
gprv13 = GaussianProcessRegressor(X, Yv13, copy(kernel))
gprω11 = GaussianProcessRegressor(X, Yω11, copy(kernel))
gprω12 = GaussianProcessRegressor(X, Yω12, copy(kernel))
gprω13 = GaussianProcessRegressor(X, Yω13, copy(kernel))
gprv21 = GaussianProcessRegressor(X, Yv21, copy(kernel))
gprv22 = GaussianProcessRegressor(X, Yv22, copy(kernel))
gprv23 = GaussianProcessRegressor(X, Yv23, copy(kernel))
gprω21 = GaussianProcessRegressor(X, Yω21, copy(kernel))
gprω22 = GaussianProcessRegressor(X, Yω22, copy(kernel))
gprω23 = GaussianProcessRegressor(X, Yω23, copy(kernel))

gprs = [gprv11, gprv12, gprv13, gprω11, gprω12, gprω13, gprv21, gprv22, gprv23, gprω21, gprω22, gprω23]
Threads.@threads for gpr in gprs
    optimize!(gpr)
end

function predictvel(gprs::Vector{GaussianProcessRegressor}, state)
    pred = Vector{Float64}(undef, 12)
    for (idx, gpr) in enumerate(gprs)
        pred[idx] = predict(gpr, state)[1][1]
    end
    return SVector{3, Float64}(pred[1:3]), SVector{3, Float64}(pred[4:6]), SVector{3, Float64}(pred[7:9]), SVector{3, Float64}(pred[10:12])  # v1, ω1, v2, ω2
end

state = getstate(mechanism)
for idx in 2:100
    global state
    v1, ω1, v2, ω2 = predictvel(gprs, state)
    projectv!([v1, v2], [ω1, ω2], mechanism)
    state = getstate(mechanism)
    overwritestorage(storage, state, idx)
end

mse = 0
for i in 1:length(data)
    mse += sum(sum((data[i][1:3]-storage.x[1][i]).^2))/3length(data)
end
println("Mean squared error: $mse")

ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
