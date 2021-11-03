using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "parallelsearch.jl"))
include(joinpath("..", "dataset.jl"))
include(joinpath("..", "dataset.jl"))


EXPERIMENT_ID = "CP_MAX"
_loadcheckpoint = false
Δtsim = 0.001
testsets = [3, 7, 9, 20]
ntrials = 1000

dataset = Dataset()
for θstart in -π:0.5:π, vstart in -2:1:2, ωstart in -2:1:2
    storage, _, _ = cartpole(Δt=Δtsim, θstart=θstart, vstart=vstart, ωstart=ωstart)
    dataset += storage
end
mechanism = cartpole(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
x_train, xnext_train, _ = sampledataset(dataset, 2500, Δt = Δtsim, exclude = testsets)
x_train = [max2mincoordinates(cstate, mechanism) for cstate in x_train]
xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]

x_train = reduce(hcat, x_train)
yv = [s[2] for s in xnext_train]
yω = [s[4] for s in xnext_train]
y_train = [yv, yω]

stdx = std(x_train, dims=2)
stdx[stdx .== 0] .= 1000
params = [100., (50 ./(stdx))...]
x_test, xnext_test, xresult_test = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
x_test = [max2mincoordinates(cstate, mechanism) for cstate in x_test]
xnext_test = [max2mincoordinates(cstate, mechanism) for cstate in xnext_test]
# intentionally not converting xresult_test since final comparison is done in maximal coordinates

paramtuples = [params .+ (4rand(length(params)) .- 1.) .* params for _ in 1:ntrials]
push!(paramtuples, params)  # Make sure initial params are also included
config = ParallelConfig(EXPERIMENT_ID, mechanism, x_train, y_train, x_test, xresult_test, paramtuples, _loadcheckpoint, xnext_test=xnext_test)

function experiment(config, params)
    l = 0.5
    mechanism = deepcopy(config.mechanism)
    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in config.y_train
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config.x_train, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(config.x_test)
        xold, vold, θold, ωold = config.x_test[i]
        xcurr, _, θcurr, _ = config.xnext_test[i]
        vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, θold, ωold], :, 1))
        xnew = xcurr + vcurr*mechanism.Δt
        θnew = θcurr + ωcurr*mechanism.Δt
        cstates = [0, xnew, zeros(12)..., 0.5l*sin(θnew)+xnew, -0.5l*cos(θnew), zeros(10)...]  # Only position matters for prediction error
        push!(predictedstates, cstates)
    end
    return predictedstates
end

function simulation(config, params)
    l = 0.5
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(300, length(mechanism.bodies))
    x, _, θ, _ = config.x_test[1]
    storage.x[1][1] = [0, x, 0]
    storage.q[1][1] = one(UnitQuaternion)
    storage.x[2][1] = [0, x+0.5l*sin(θ), -0.5l*cos(θ)]
    storage.q[2][1] = UnitQuaternion(RotX(θ))

    gps = Vector()
    for yi in config.y_train
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config.x_train, yi, MeanZero(), kernel)
        # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    xold, vold, θold, ωold = config.x_test[1]
    xcurr, _, θcurr = config.xnext_test[1]
    for i in 2:length(storage.x[1])
        vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, θold, ωold], :, 1))
        storage.x[1][i] = [0, xcurr, 0]
        storage.q[1][i] = one(UnitQuaternion)
        storage.x[2][i] = [0, xcurr+0.5l*sin(θcurr), -0.5l*cos(θcurr)]
        storage.q[2][i] = UnitQuaternion(RotX(θcurr))
        xold, vold, θold, ωold = xcurr, vcurr, θcurr,ωcurr
        xcurr += vcurr*mechanism.Δt
        θcurr += ωcurr*mechanism.Δt
    end
    return storage
end

#storage = simulation(config, params)
#ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
parallelsearch(experiment, config)