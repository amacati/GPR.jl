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


EXPERIMENT_ID = "P1_MIN"
_loadcheckpoint = false
Δtsim = 0.001
testsets = [3, 7, 9, 20]
ntrials = 1

dataset = Dataset()
for θ in -π/2:0.1:π/2
    storage, _, _ = simplependulum2D(Δt=Δtsim, θstart=θ)
    dataset += storage
end
mechanism = simplependulum2D(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
x_train, xnext_train, _ = sampledataset(dataset, 2500, Δt = Δtsim, exclude = testsets)
x_train = [max2mincoordinates(cstate, mechanism) for cstate in x_train]
xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]
cleardata!((x_train, xnext_train), ϵ = 1e-5)

x_train = reduce(hcat, x_train)
y_train = [[s[2] for s in xnext_train]]

stdx = std(x_train, dims=2)
stdx[stdx .== 0] .= 100
params = [1.1, (10 ./(stdx))...]
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
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(config.x_train, config.y_train[1], MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))

    for i in 1:length(config.x_test)
        θold, ωold = config.x_test[i]
        θcurr, _ = config.xnext_test[i]
        ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
        θnew = θcurr + ωcurr*mechanism.Δt
        q1 = UnitQuaternion(RotX(θnew))
        vq1 = [q1.w, q1.x, q1.y, q1.z]
        cstate = [0, 0.5sin(θnew), -0.5cos(θnew), vq1..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates
end

function simulation(config, params)
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(300, length(mechanism.bodies))
    θ = config.x_test[1][1]
    storage.x[1][1] = [0, 0.5sin(θ), -0.5cos(θ)]
    storage.q[1][1] = UnitQuaternion(RotX(θ))

    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(config.x_train, config.y_train[1], MeanZero(), kernel)
    # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))

    θold, ωold = config.x_test[1]
    θcurr, _ = config.xnext_test[1]
    for i in 2:length(storage.x[1])
        ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
        storage.x[1][i] = [0, 0.5sin(θcurr), -0.5cos(θcurr)]
        storage.q[1][i] = UnitQuaternion(RotX(θcurr))
        θold, ωold = θcurr, ωcurr
        θcurr = θcurr + ωcurr*mechanism.Δt  # ω*Δt
    end
    return storage
end

# storage = simulation(config, params)
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
parallelsearch(experiment, config)
