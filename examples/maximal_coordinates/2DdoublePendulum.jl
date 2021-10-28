using GPR
using ConstrainedDynamicsVis
using ConstrainedDynamics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))


EXPERIMENT_ID = "P2_2D_MAX_GGK"
_loadcheckpoint = false
paramtuples = collect(Iterators.product(0.1:0.5:15.1, 0.1:0.5:15.1, 0.1:0.5:15.1))

storage, mechanism, initialstates = doublependulum2D(noise = true)
data = loaddata(storage)
cleardata!(data)

X = data[1:end-1]
Yv11 = [s[8] for s in data[2:end]]
Yv12 = [s[9] for s in data[2:end]]
Yv13 = [s[10] for s in data[2:end]]
Yv21 = [s[21] for s in data[2:end]]
Yv22 = [s[22] for s in data[2:end]]
Yv23 = [s[23] for s in data[2:end]]
Yω11 = [s[11] for s in data[2:end]]
Yω12 = [s[12] for s in data[2:end]]
Yω13 = [s[13] for s in data[2:end]]
Yω21 = [s[24] for s in data[2:end]]
Yω22 = [s[25] for s in data[2:end]]
Yω23 = [s[26] for s in data[2:end]]
Y = [Yv11, Yv12, Yv13, Yv21, Yv22, Yv23, Yω11, Yω12, Yω13, Yω21, Yω22, Yω23]

config = ParallelConfig(EXPERIMENT_ID, mechanism, storage, X, Y, paramtuples, _loadcheckpoint)

#pkernel = GeneralGaussianKernel(15, ones(3)*0.001)
#qkernel = QuaternionKernel(15, ones(3)*0.001)
#vkernel = GeneralGaussianKernel(15, ones(6)*0.001)
# GGK(10, 0.001), QK(10, 0.001), GGK(10, 0.001) *2 works as well
#kernel = CompositeKernel([copy(pkernel), copy(qkernel), copy(vkernel), copy(pkernel), copy(qkernel), copy(vkernel)], [3, 4, 6, 3, 4, 6])

function experiment(config, params)
    mechanism = deepcopy(config.mechanism)
    storage = Storage{Float64}(length(config.storage.x[1]), length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = config.storage.x[id][1]
        storage.q[id][1] = config.storage.q[id][1]
        storage.v[id][1] = config.storage.v[id][1]
        storage.ω[id][1] = config.storage.ω[id][1]
    end
    kernel = GeneralGaussianKernel(params[1], [params[2:end]...])  # 500., 0.001 works okayish
    gprs = Vector{GaussianProcessRegressor}()
    for Yi in config.Y
        push!(gprs, GaussianProcessRegressor(X, Yi, copy(kernel)))
    end
    mogpr = MOGaussianProcessRegressor(gprs)
    # optimize!(mogpr, verbose=false)

    for i in 2:length(storage.x[1])-1
        oldstates = getstates(config.storage, i-1)
        setstates!(mechanism, oldstates)
        μ = GPR.predict(mogpr, [SVector(reduce(vcat, oldstates)...)])[1][1]
        vcurr, ωcurr = [SVector(μ[1:3]...), SVector(μ[4:6]...)], [SVector(μ[7:9]...), SVector(μ[10:12]...)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        states = getstates(mechanism)  # Extract xnew
        overwritestorage(storage, states, i+1)
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
    kernel = GeneralGaussianKernel(params[1], [params[2:end]...])  # 500., 0.001 works okayish
    gprs = Vector{GaussianProcessRegressor}()
    for Yi in config.Y
        push!(gprs, GaussianProcessRegressor(X, Yi, copy(kernel)))
    end
    mogpr = MOGaussianProcessRegressor(gprs)
    optimize!(mogpr, verbose=false)

    states = getstates(config.storage, 1)
    setstates!(mechanism, states)
    for i in 2:length(storage.x[1])
        μ = GPR.predict(mogpr, [SVector(reduce(vcat, states)...)])[1][1]
        vcurr, ωcurr = [SVector(μ[1:3]...), SVector(μ[4:6]...)], [SVector(μ[7:9]...), SVector(μ[10:12]...)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getstates(mechanism)
        overwritestorage(storage, states, i)
    end
    return storage
end

storage = simulation(config, [500., ones(26)...])
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
