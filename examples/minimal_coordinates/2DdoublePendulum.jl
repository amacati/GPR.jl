using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))


EXPERIMENT_ID = "P2_2D_MIN_GGK"
_loadcheckpoint = false

storage, mechanism, initialstates = doublependulum2D(noise = true)
data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
data = [SVector(s[1], s[2], s[3], s[4]) for s in data]
cleardata!(data)
X = reduce(hcat, data[1:end-1])
stdx = std(X, dims=2)
stdx[stdx .== 0] .= 100
params = [1.1, (1 ./(0.02 .*stdx))...]
paramtuples = [params]

Yω1 = [s[2] for s in data[2:end]]
Yω2 = [s[4] for s in data[2:end]]

config = ParallelConfig(EXPERIMENT_ID, mechanism, storage, X, [Yω1, Yω2], paramtuples, _loadcheckpoint)

function experiment(config, params)
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(length(config.storage.x[1]), length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = config.storage.x[id][1]
        storage.q[id][1] = config.storage.q[id][1]
        storage.v[id][1] = config.storage.v[id][1]
        storage.ω[id][1] = config.storage.ω[id][1]
    end
    gps = Vector()
    for Yi in config.Y
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config.X, Yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking()), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    l2 = sqrt(2) / 2
    for i in 2:length(storage.x[1])-1
        θ1old, ω1old, θ2old, ω2old = max2mincoordinates([vcat(getstates(config.storage, i-1)...)], mechanism)[1]
        θ1curr, _, θ2curr, _ = max2mincoordinates([vcat(getstates(config.storage, i)...)], mechanism)[1]
        ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
        θ1new = θ1curr + ω1curr*mechanism.Δt  # ω1*Δt
        θ2new = θ2curr + ω2curr*mechanism.Δt  # ω2*Δt
        storage.x[1][i+1] = [0, 0.5sin(θ1new), -0.5cos(θ1new)]
        storage.q[1][i+1] = UnitQuaternion(RotX(θ1new))
        storage.x[2][i+1] = [0, sin(θ1new) + 0.5l2*sin(θ1new+θ2new), -cos(θ1new) - 0.5l2*cos(θ1new + θ2new)]
        storage.q[2][i+1] = UnitQuaternion(RotX(θ1new + θ2new))
    end
    return storage
end

function simulation(config, params)
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(length(config.storage.x[1]), length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = config.storage.x[id][1]
        storage.q[id][1] = config.storage.q[id][1]
        storage.v[id][1] = config.storage.v[id][1]
        storage.ω[id][1] = config.storage.ω[id][1]
    end
    gps = Vector()
    for Yi in config.Y
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config.X, Yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking()), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    l2 = sqrt(2) / 2
    θ1old, ω1old, θ2old, ω2old = max2mincoordinates([vcat(getstates(config.storage, 1)...)], mechanism)[1]
    θ1curr, _, θ2curr, _ = max2mincoordinates([vcat(getstates(config.storage, 2)...)], mechanism)[1]
    for i in 2:length(storage.x[1])-1
        ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
        θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
        θ1new = θ1curr + ω1curr*mechanism.Δt  # ω1*Δt
        θ2new = θ2curr + ω2curr*mechanism.Δt  # ω2*Δt
        storage.x[1][i+1] = [0, 0.5sin(θ1new), -0.5cos(θ1new)]
        storage.q[1][i+1] = UnitQuaternion(RotX(θ1new))
        storage.x[2][i+1] = [0, sin(θ1new) + 0.5l2*sin(θ1new+θ2new), -cos(θ1new) - 0.5l2*cos(θ1new + θ2new)]
        storage.q[2][i+1] = UnitQuaternion(RotX(θ1new + θ2new))
        θ1curr, θ2curr = θ1new, θ2new
    end
    return storage
end

storage = simulation(config, params)
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
# parallelsearch(experiment, config)