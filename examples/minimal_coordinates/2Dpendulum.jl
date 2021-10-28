using GPR
using ConstrainedDynamicsVis
using ConstrainedDynamics
using Plots
using StatsBase: sample

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))


EXPERIMENT_ID = "P1_2D_MIN_GGK"
_loadcheckpoint = false
paramtuples = collect(Iterators.product(10.1:0.5:11.1, 10.1:0.5:11.1, 10.1:0.5:11.1))# collect(Iterators.product(0.1:0.5:15.1, 0.1:0.5:15.1, 0.1:0.5:15.1))

storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
data = [SVector(state[1], state[2]) for state in data]
cleardata!(data)
X = data[1:end-1]
Yω = [s[2] for s in data[2:end]]

config = ParallelConfig(EXPERIMENT_ID, deepcopy(mechanism), X, [Yω], initialstates, length(storage.x[1]), paramtuples, _loadcheckpoint)

function experiment(config, params)
    mechanism = config.mechanism
    storage = Storage{Float64}(config.experimentlength, length(mechanism.bodies))
    kernel = GeneralGaussianKernel(params[1], [params[2:3]...])
    gprω = GaussianProcessRegressor(config.X, config.Y[1], kernel)
    optimize!(gprω)

    resetMechanism!(mechanism, config.initialstates)
    foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
    maxstates = getstates(mechanism)
    states = max2mincoordinates(maxstates, mechanism)[1]
    θ, ω = states[1], states[2]
    for i in 2:config.experimentlength
        ω = GPR.predict(gprω, SVector(θ, ω))[1][1]
        θ += ω*mechanism.Δt
        storage.x[1][i] = [0, 0.5sin(θ), -0.5cos(θ)]
        storage.q[1][i] = UnitQuaternion(RotX(θ))
    end
    return storage
end

parallelsearch(experiment, config)