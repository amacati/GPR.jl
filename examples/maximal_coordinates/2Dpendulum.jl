using GPR
using GaussianProcesses
using ConstrainedDynamicsVis
using ConstrainedDynamics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))


EXPERIMENT_ID = "P1_2D_MAX_GGK"
_loadcheckpoint = false
Λmin, Λmax, ΛN = 0.1, 15.1, 2
paramtuples = collect(Iterators.product(range(Λmin, stop=Λmax, length=ΛN), range(Λmin, stop=Λmax, length=ΛN)))

storage, mechanism, initialstates = simplependulum2D(noise = true)
data = loaddata(storage)
cleardata!(data)
X = reduce(hcat, data[1:end-1])
Yv1 = [s[8] for s in data[2:end]]
Yv2 = [s[9] for s in data[2:end]]
Yv3 = [s[10] for s in data[2:end]]
Yω1 = [s[11] for s in data[2:end]]
Yω2 = [s[12] for s in data[2:end]]
Yω3 = [s[13] for s in data[2:end]]
Y = [Yv1, Yv2, Yv3, Yω1, Yω2, Yω3]

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
        kernel = SEArd(ones(13)*log(params[2]), log(params[1]))
        gp = GP(config.X, Yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp)
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
        states = getstates(mechanism)  # Extract xnew
        overwritestorage(storage, states, i+1)
    end
    return storage
end

# storage = experiment(config, [10.5, ones(13)*10...])
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
parallelsearch(experiment, config)