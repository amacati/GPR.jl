using GPR
using ConstrainedDynamicsVis
using ConstrainedDynamics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))


EXPERIMENT_ID = "P2_2D_MIN_GGK"
_loadcheckpoint = false
paramtuples = collect(Iterators.product(0.1:2:15.1, 0.1:2:15.1, 0.1:2:15.1, 0.1:2:15.1, 0.1:2:15.1))

storage, mechanism, initialstates = doublependulum2D(noise = true)
data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
data = [SVector(s[1], s[2], s[3], s[4]) for s in data]
cleardata!(data)
X = data[1:end-1]
Yω1 = [s[2] for s in data[2:end]]
Yω2 = [s[4] for s in data[2:end]]

config = ParallelConfig(EXPERIMENT_ID, deepcopy(mechanism), storage, X, [Yω1, Yω2], initialstates, length(storage.x[1]), paramtuples, _loadcheckpoint)

function experiment(config, params)
    mechanism = config.mechanism
    storage = Storage{Float64}(config.experimentlength, length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = config.storage.x[id][1]
        storage.q[id][1] = config.storage.q[id][1]
        storage.v[id][1] = config.storage.v[id][1]
        storage.ω[id][1] = config.storage.ω[id][1]
    end
    kernel = GeneralGaussianKernel(params[1], [params[2:5]...])
    gprω1 = GaussianProcessRegressor(config.X, config.Y[1], copy(kernel))
    gprω2 = GaussianProcessRegressor(config.X, config.Y[2], copy(kernel))
    mogpr = MOGaussianProcessRegressor([gprω1, gprω2])
    optimize!(mogpr)

    l2 = sqrt(2) / 2
    for i in 2:config.experimentlength
        maxstates = getstates(config.storage, i)
        states = max2mincoordinates([vcat(maxstates...)], mechanism)[1]
        θ1, ω1, θ2, ω2 = states
        μ = GPR.predict(mogpr, SVector(θ1, ω1, θ2, ω2))[1][1]
        θ1 += μ[1]*mechanism.Δt  # ω1*Δt
        θ2 += μ[2]*mechanism.Δt  # ω2*Δt
        storage.x[1][i] = [0, 0.5sin(θ1), -0.5cos(θ1)]
        storage.q[1][i] = UnitQuaternion(RotX(θ1))
        storage.x[2][i] = [0, sin(θ1) + 0.5l2*sin(θ1+θ2), -cos(θ1) - 0.5l2*cos(θ1 + θ2)]
        storage.q[2][i] = UnitQuaternion(RotX(θ1 + θ2))
    end
    return storage
end

parallelsearch(experiment, config)