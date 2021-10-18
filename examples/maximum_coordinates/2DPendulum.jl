using Statistics
using GPR
using Plots
using LinearAlgebra
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage)
resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position
steps = 30
X = data[1:steps:end-1]
Yv1 = [sample[8] for sample in data[2:steps:end]]
Yv2 = [sample[9] for sample in data[2:steps:end]]
Yv3 = [sample[10] for sample in data[2:steps:end]]
Yω1 = [sample[11] for sample in data[2:steps:end]]
Yω2 = [sample[12] for sample in data[2:steps:end]]
Yω3 = [sample[13] for sample in data[2:steps:end]]

kernel = GaussianKernel(0.5, 1.5)
gprv1 = GaussianProcessRegressor(X, Yv1, copy(kernel))
gprv2 = GaussianProcessRegressor(X, Yv2, copy(kernel))
gprv3 = GaussianProcessRegressor(X, Yv3, copy(kernel))
gprω1 = GaussianProcessRegressor(X, Yω1, copy(kernel))
gprω2 = GaussianProcessRegressor(X, Yω2, copy(kernel))
gprω3 = GaussianProcessRegressor(X, Yω3, copy(kernel))
gprs = [gprv1, gprv2, gprv3, gprω1, gprω2, gprω3]
for gpr in gprs
    optimize!(gpr)
end

function predictvel(gprs::Vector{GaussianProcessRegressor}, state)
    pred = Vector{Float64}(undef, 6)
    for (idx, gpr) in enumerate(gprs)
        pred[idx] = predict(gpr, state)[1][1]
    end
    return SVector{3, Float64}(pred[1:3]), SVector{3, Float64}(pred[4:6])  # v and ω
end

state = getstate(mechanism)
project = false
for idx in 2:1000
    global state
    v, ω = predictvel(gprs, state)
    if project
        vs, ωs = projectv!([v], [ω], mechanism)
        v, ω = vs[1], ωs[1]
    else
        s = Vector{Float64}()
        append!(s, v)
        append!(s, ω)
        for _ in 1:sum([length(eqc) for eqc in mechanism.eqconstraints]) push!(s,0.) end
        updateMechanism!(mechanism, s)
        foreach(ConstrainedDynamics.updatestate!, mechanism.bodies, mechanism.Δt)
    end
    foreach(ConstrainedDynamics.updatesolution!, mechanism.bodies)
    foreach(ConstrainedDynamics.updatesolution!, mechanism.eqconstraints)
    state = getstate(mechanism)
    overwritestorage(storage, state, idx)
end

mse = 0
for i in 1:length(data)
    mse += sum(sum((data[i][1:3]-storage.x[1][i]).^2))/3length(data)
end
println("Mean squared error: $mse")

ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
