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


EXPERIMENT_ID = "P1_MIN"
_loadcheckpoint = false

storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
data = [SVector(s[1], s[2]) for s in data]
cleardata!(data, ϵ=1e-4)
display(length(data))
X = reduce(hcat, data[1:end-1])
Yω = [s[2] for s in data[2:end]]

stdx = std(X, dims=2)
stdx[stdx .== 0] .= 100
params = [1.1, (10 ./(stdx))...]
# params = [0.1, ones(2).*10...]
paramtuples = [params]

config = ParallelConfig(EXPERIMENT_ID, mechanism, storage, X, [Yω], paramtuples, _loadcheckpoint)

function experiment(config, params)
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(length(config.storage.x[1]), length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = config.storage.x[id][1]
        storage.q[id][1] = config.storage.q[id][1]
        storage.v[id][1] = config.storage.v[id][1]
        storage.ω[id][1] = config.storage.ω[id][1]
    end
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(config.X, config.Y[1], MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking()), Optim.Options(time_limit=10.))

    for i in 2:length(config.storage.x[1])-1
        θold, ωold = max2mincoordinates([vcat(getstates(config.storage, i-1)...)], mechanism)[1]
        θcurr, _ = max2mincoordinates([vcat(getstates(config.storage, i)...)], mechanism)[1]
        ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
        θnew = θcurr + ωcurr*mechanism.Δt
        storage.x[1][i+1] = [0, 0.5sin(θnew), -0.5cos(θnew)]
        storage.q[1][i+1] = UnitQuaternion(RotX(θnew))
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
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(config.X, config.Y[1], MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking()), Optim.Options(time_limit=10.))

    θold, ωold = max2mincoordinates([vcat(getstates(config.storage, 1)...)], mechanism)[1]
    θcurr, _ = max2mincoordinates([vcat(getstates(config.storage, 2)...)], mechanism)[1]
    for i in 2:length(config.storage.x[1])-1
        ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
        θold, ωold = θcurr, ωcurr
        θnew = θcurr + ωcurr*mechanism.Δt  # ω*Δt
        storage.x[1][i+1] = [0, 0.5sin(θnew), -0.5cos(θnew)]
        storage.q[1][i+1] = UnitQuaternion(RotX(θnew))
        θcurr = θnew
    end
    return storage
end

# storage = simulation(config, [10.1, 10.1, 10.1])
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
parallelsearch(experiment, config)
