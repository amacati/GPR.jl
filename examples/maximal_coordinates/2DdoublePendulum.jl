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
include(joinpath("..", "dataset.jl"))


EXPERIMENT_ID = "P2_MAX"
_loadcheckpoint = false
Δtsim = 0.001
testsets = [3, 7, 9, 20]
ntrials = 0
ntrainingsamples = 0

dataset = Dataset()
for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
    storage, _, _ = doublependulum2D(Δt=Δtsim, θstart=[θ1, θ2])
    dataset += storage
end
x_train, xnext_train, _ = sampledataset(dataset, 50, Δt = Δtsim, exclude = testsets)

x_train = reduce(hcat, x_train)
yv12 = [s[9] for s in xnext_train]
yv13 = [s[10] for s in xnext_train]
yv22 = [s[22] for s in xnext_train]
yv23 = [s[23] for s in xnext_train]
yω11 = [s[11] for s in xnext_train]
yω21 = [s[24] for s in xnext_train]
y_train = [yv12, yv13, yv22, yv23, yω11, yω21]

stdx = std(x_train, dims=2)
stdx[stdx .== 0] .= 100
params = [1.1, (1 ./(0.02 .*stdx))...]
x_test, _, xresult_test = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])

mechanism = doublependulum2D(Δt=0.01, θstart=[0, 0])[2]  # Reset Δt to 0.01 in mechanism
paramtuples = [params .+ (4rand(length(params)) .- 1.) .* params for _ in 1:ntrials]
push!(paramtuples, params)  # Make sure initial params are also included
config = ParallelConfig(EXPERIMENT_ID, mechanism, x_train, y_train, x_test, xresult_test, paramtuples, _loadcheckpoint)

function experiment(config, params)
    mechanism = deepcopy(config.mechanism)
    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in config.y_train
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config.x_train, yi, MeanZero(), kernel)
        # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(config.x_test)
        oldstates = tovstate(config.x_test[i])
        setstates!(mechanism, oldstates)
        μ = predict_velocities(gps, reshape(reduce(vcat, oldstates), :, 1))
        vcurr, ωcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...)], [SVector(μ[5], 0, 0), SVector(μ[6], 0, 0)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates
end

function simulation(config, params)
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(300, length(mechanism.bodies))
    initialstates = tovstate(config.x_test[1])
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = initialstates[id][1:3]
        storage.q[id][1] = UnitQuaternion(initialstates[id][4], initialstates[id][5:7])
        storage.v[id][1] = initialstates[id][8:10]
        storage.ω[id][1] = initialstates[id][11:13]
    end

    gps = Vector()
    for yi in config.y_train
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config.x_train, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    states = tovstate(config.x_test[1])
    setstates!(mechanism, states)
    for i in 2:length(storage.x[1])
        μ = predict_velocities(gps, reshape(reduce(vcat, states), :, 1))
        vcurr, ωcurr = [SVector(μ[1:3]...), SVector(μ[4:6]...)], [SVector(μ[7:9]...), SVector(μ[10:12]...)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getvstates(mechanism)
        overwritestorage(storage, states, i)
    end
    return storage
end

# storage = simulation(config, params)
# storage, mechanism, initialstates = doublependulum2D(Δt=0.01, θstart=[-π/3, collect(-π/3:0.5:π/3)[3]])
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
parallelsearch(experiment, config)
