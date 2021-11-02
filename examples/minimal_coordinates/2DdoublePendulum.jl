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


EXPERIMENT_ID = "P2_MIN"
_loadcheckpoint = false
Δtsim = 0.001
testsets = [3, 7, 9, 20]
ntrials = 1000

dataset = Dataset()
for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
    storage, _, _ = doublependulum2D(Δt=Δtsim, θstart=[θ1, θ2])
    dataset += storage
end
mechanism = doublependulum2D(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
x_train, xnext_train, _ = sampledataset(dataset, 2500, Δt = Δtsim, exclude = testsets)
x_train = [max2mincoordinates(cstate, mechanism) for cstate in x_train]
xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]
cleardata!((x_train, xnext_train), ϵ = 1e-4)

x_train = reduce(hcat, x_train)
yω1 = [s[2] for s in xnext_train]
yω2 = [s[4] for s in xnext_train]
y_train = [yω1, yω2]

stdx = std(x_train, dims=2)
stdx[stdx .== 0] .= 100
params = [1.1, (1 ./(0.02 .*stdx))...]
x_test, xnext_test, xresult_test = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testset)])
x_test = [max2mincoordinates(cstate, mechanism) for cstate in x_test]
xnext_test = [max2mincoordinates(cstate, mechanism) for cstate in xnext_test]
# intentionally not converting xresult_test since final comparison is done in maximal coordinates

paramtuples = [params .+ (4rand(length(params)) .- 1.) .* params for _ in 1:ntrials]
push!(paramtuples, params)  # Make sure initial params are also included
config = ParallelConfig(EXPERIMENT_ID, mechanism, x_train, y_train, x_test, xresult_test, paramtuples, _loadcheckpoint, xnext_test=xnext_test)

function experiment(config, params)
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

    l2 = sqrt(2) / 2
    for i in 1:length(config.x_test)
        θ1old, ω1old, θ2old, ω2old = config.x_test[i]
        θ1curr, _, θ2curr, _ = config.xnext_test[i]
        ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
        θ1new = θ1curr + ω1curr*mechanism.Δt  # ω1*Δt
        θ2new = θ2curr + ω2curr*mechanism.Δt  # ω2*Δt
        q1, q2 = UnitQuaternion(RotX(θ1new)), UnitQuaternion(RotX(θ1new + θ2new))
        vq1, vq2 = [q1.w, q1.x, q1.y, q1.z], [q2.w, q2.x, q2.y, q2.z]
        cstates = [0, 0.5sin(θ1new), -0.5cos(θ1new), vq1..., zeros(6)...,
                   0, sin(θ1new) + 0.5l2*sin(θ1new+θ2new), -cos(θ1new) - 0.5l2*cos(θ1new + θ2new), vq2..., zeros(6)...]
        push!(predictedstates, cstates)
    end
    return predictedstates
end

function simulation(config, params)
    l2 = sqrt(2) / 2  # Length param of the second pendulum link
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(300, length(mechanism.bodies))
    θ1, _, θ2, _ = config.x_test[1]
    storage.x[1][1] = [0, 0.5sin(θ1), -0.5cos(θ1)]
    storage.q[1][1] = UnitQuaternion(RotX(θ1))
    storage.x[2][1] = [0, sin(θ1) + 0.5l2*sin(θ1+θ2), -cos(θ1) - 0.5l2*cos(θ1 + θ2)]
    storage.q[2][1] = UnitQuaternion(RotX(θ1 + θ2))

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

    θ1old, ω1old, θ2old, ω2old = config.x_test[1]
    θ1curr, _, θ2curr, _ = config.xnext_test[1]
    for i in 2:length(storage.x[1])
        ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
        storage.x[1][i] = [0, 0.5sin(θ1curr), -0.5cos(θ1curr)]
        storage.q[1][i] = UnitQuaternion(RotX(θ1curr))
        storage.x[2][i] = [0, sin(θ1curr) + 0.5l2*sin(θ1curr+θ2curr), -cos(θ1curr) - 0.5l2*cos(θ1curr + θ2curr)]
        storage.q[2][i] = UnitQuaternion(RotX(θ1curr + θ2curr))
        θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
        θ1curr = θ1curr + ω1curr*mechanism.Δt  # ω1*Δt
        θ2curr = θ2curr + ω2curr*mechanism.Δt  # ω2*Δt
    end
    return storage
end

# storage = simulation(config, params)
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
parallelsearch(experiment, config)