using GPR
using LinearAlgebra
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


storage, mechanism, initialstates = simplependulum3D()
data = loaddata(storage)
resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position
steps = 11
X = data[2:steps:end-1]
Yv1 = [sample[8] for sample in data[3:steps:end]]
Yv2 = [sample[9] for sample in data[3:steps:end]]
Yv3 = [sample[10] for sample in data[3:steps:end]]
Yω1 = [sample[11] for sample in data[3:steps:end]]
Yω2 = [sample[12] for sample in data[3:steps:end]]
Yω3 = [sample[13] for sample in data[3:steps:end]]

# kernel = GaussianKernel(0.5, 0.2)
kernel = GeneralGaussianKernel(0.5, ones(13)*1.5)  # 4., 0.5
gprv1 = GaussianProcessRegressor(X, Yv1, copy(kernel))
gprv2 = GaussianProcessRegressor(X, Yv2, copy(kernel))
gprv3 = GaussianProcessRegressor(X, Yv3, copy(kernel))
gprω1 = GaussianProcessRegressor(X, Yω1, copy(kernel))
gprω2 = GaussianProcessRegressor(X, Yω2, copy(kernel))
gprω3 = GaussianProcessRegressor(X, Yω3, copy(kernel))
gprs = [gprv1, gprv2, gprv3, gprω1, gprω2, gprω3]
Threads.@threads for gpr in gprs
    optimize!(gpr, verbose=true)
end

function predictvel(gprs::Vector{GaussianProcessRegressor}, state)
    pred = Vector{Float64}(undef, 6)
    for (idx, gpr) in enumerate(gprs)
        pred[idx] = predict(gpr, state)[1][1]
    end
    return SVector{3, Float64}(pred[1:3]), SVector{3, Float64}(pred[4:6])  # v and ω
end

state = getstate(mechanism)
for idx in 2:1000
    global state
    v, ω = predictvel(gprs, state)
    projectv!([v], [ω], mechanism)
    state = getstate(mechanism)
    overwritestorage(storage, state, idx)
end

mse = 0
for i in 1:length(data)
    mse += sum(sum((data[i][1:3]-storage.x[1][i]).^2))/3length(data)
end
println("Mean squared error: $mse")  # best: Λ=1.5, steps: 11, MSE: 3.625022069393289e-6

ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
