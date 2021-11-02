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


EXPERIMENT_ID = "P1_MAX"
_loadcheckpoint = false

storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage)
cleardata!(data, ϵ = 1e-4)

X = reduce(hcat, data[1:end-1])
Yv1 = [s[8] for s in data[2:end]]
Yv2 = [s[9] for s in data[2:end]]
Yv3 = [s[10] for s in data[2:end]]
Yω1 = [s[11] for s in data[2:end]]
Yω2 = [s[12] for s in data[2:end]]
Yω3 = [s[13] for s in data[2:end]]
Y = [Yv1, Yv2, Yv3, Yω1, Yω2, Yω3]

stdx = std(X, dims=2)
stdx[stdx .== 0] .= 1000
params = [100., (10 ./(stdx))...]
display(params)
paramtuples = [params]
config = ParallelConfig(EXPERIMENT_ID, mechanism, storage, X, Y, paramtuples, _loadcheckpoint)

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

    for i in 2:length(storage.x[1])-1
        oldstates = getstates(config.storage, i-1)
        setstates!(mechanism, oldstates)
        μ = predict_velocities(gps, reshape(reduce(vcat, oldstates), :, 1))
        vcurr, ωcurr = [SVector(μ[1:3]...)], [SVector(μ[4:6]...)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        overwritestorage(storage, getstates(mechanism), i+1)  # Write xnew to storage
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
        # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking()), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    

    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    states = getstates(config.storage, 1)
    setstates!(mechanism, states)
    for i in 2:length(storage.x[1])
        μ = predict_velocities(gps, reshape(reduce(vcat, states), :, 1))
        vcurr, ωcurr = [SVector(μ[1:3]...)], [SVector(μ[4:6]...)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getstates(mechanism)
        overwritestorage(storage, states, i)
    end
    return storage
end

storage = simulation(config, params)
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
# arallelsearch(experiment, config)