using GPR
using ConstrainedDynamicsVis
using ConstrainedDynamics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))


EXPERIMENT_ID = "P1_2D_MIN_GGK"
_loadcheckpoint = false
paramtuples = collect(Iterators.product(0.1:0.5:15.1, 0.1:0.5:15.1, 0.1:0.5:15.1))
# paramtuples = collect(Iterators.product(10.1:0.5:11.1, 10.1:0.5:11.1, 10.1:0.5:11.1))

storage, mechanism, initialstates = simplependulum2D(noise = true)
data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
data = [SVector(state[1], state[2]) for state in data]
cleardata!(data)
X = data[1:end-1]
Yω = [s[2] for s in data[2:end]]

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
    kernel = GeneralGaussianKernel(params[1], [params[2:3]...])
    gprω = GaussianProcessRegressor(config.X, config.Y[1], kernel)
    optimize!(gprω)

    for i in 2:length(config.storage.x[1])-1
        θold, ωold = max2mincoordinates([vcat(getstates(config.storage, i-1)...)], mechanism)[1]
        θcurr, _ = max2mincoordinates([vcat(getstates(config.storage, i)...)], mechanism)[1]
        ωcurr = GPR.predict(gprω, SVector(θold, ωold))[1][1]
        θnew = θcurr + ωcurr*mechanism.Δt
        storage.x[1][i+1] = [0, 0.5sin(θnew), -0.5cos(θnew)]
        storage.q[1][i+1] = UnitQuaternion(RotX(θnew))
    end
    return storage
end

# storage = experiment(config, [10.1, 10.1, 10.1])
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
parallelsearch(experiment, config)
