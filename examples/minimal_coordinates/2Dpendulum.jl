using GPR
using ConstrainedDynamicsVis
using ConstrainedDynamics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))


EXPERIMENT_ID = "P1_2D_MIN_GGK"
_loadcheckpoint = false
paramtuples = collect(Iterators.product(10.1:0.5:11.1, 10.1:0.5:11.1, 10.1:0.5:11.1))# collect(Iterators.product(0.1:0.5:15.1, 0.1:0.5:15.1, 0.1:0.5:15.1))

storage, mechanism, initialstates = simplependulum2D(noise = true)
data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
data = [SVector(state[1], state[2]) for state in data]
cleardata!(data)
X = data[1:end-1]
Yω = [s[2] for s in data[2:end]]

config = ParallelConfig(EXPERIMENT_ID, deepcopy(mechanism), storage, X, [Yω], initialstates, length(storage.x[1]), paramtuples, _loadcheckpoint)

function experiment(config, params)
    mechanism = config.mechanism
    storage = Storage{Float64}(config.experimentlength, length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = config.storage.x[id][1]
        storage.q[id][1] = config.storage.q[id][1]
        storage.v[id][1] = config.storage.v[id][1]
        storage.ω[id][1] = config.storage.ω[id][1]
    end
    kernel = GeneralGaussianKernel(params[1], [params[2:3]...])
    gprω = GaussianProcessRegressor(config.X, config.Y[1], kernel)
    optimize!(gprω)

    for i in 2:config.experimentlength
        maxstates = getstates(config.storage, i)
        states = max2mincoordinates([vcat(maxstates...)], mechanism)[1]
        θ, ω = states
        ω = GPR.predict(gprω, SVector(θ, ω))[1][1]
        θ += ω*mechanism.Δt
        storage.x[1][i] = [0, 0.5sin(θ), -0.5cos(θ)]
        storage.q[1][i] = UnitQuaternion(RotX(θ))
    end
    return storage
end

parallelsearch(experiment, config)