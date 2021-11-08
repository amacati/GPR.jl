using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentFBMin(config)
    mechanism = deepcopy(config["mechanism"])
    l = mechanism.bodies[1].shape.xyz[3]
    # Sample from dataset
    dataset = config["dataset"]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    xtrain_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t0]
    xtrain_t0 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_t0]
    xtrain_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t1]
    xtrain_t1 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_t1]
    xtrain_t0 = reduce(hcat, xtrain_t0)
    ω1 = [s[2] for s in xtrain_t1]
    ω2 = [s[4] for s in xtrain_t1]
    ytrain = [ω1, ω2]
    xtest_t0, xtest_t1, xtest_tk = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, 
                                                 pseudorandom = true, exclude = trainsets, stepsahead=[0,1,config["simsteps"]+1])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]
    xtest_t0 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtest_t0]
    xtest_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1]
    xtest_t1 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtest_t1]
    # intentionally not converting xtest_tk since final comparison is done in maximal coordinates

    stdx = std(xtrain_t0, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1., (10 ./(stdx))...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_t0, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_t0)
        θ1old, ω1old, θ2old, ω2old = xtest_t0[i]
        θ1curr, _, θ2curr, _ = xtest_t1[i]
        for _ in 1:config["simsteps"]
            ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
            θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
            θ1curr = θ1curr + ω1curr*mechanism.Δt
            θ2curr = θ2curr + ω2curr*mechanism.Δt
        end
        x1 = [0, 0.5sin(θ1curr), -0.5cos(θ1curr)]
        x2 = [0, sin(θ1curr) + 0.5sin(θ2curr), -cos(θ1curr) - 0.5cos(θ2curr)]
        x3 = [0, 0.5sin(θ2curr), -0.5cos(θ2curr)]
        x4 = [0, sin(θ2curr) + 0.5sin(θ1curr), -cos(θ2curr) - 0.5cos(θ1curr)]
        cstate = [x1..., zeros(10)..., x2..., zeros(10)..., x3..., zeros(10)..., x4..., zeros(10)...]  # Orientation, velocities not used in error
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_tk, params
end

function simulation()
    dataset = Dataset()
    _storage, mechanism, initialstates = fourbar(θstart = [π/4, π/4])
    l = mechanism.bodies[1].shape.xyz[3]
    dataset += _storage
    xtrain_t0, xtrain_t1 = sampledataset(dataset, 200, random = true, replace=false)
    xtrain_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t0]
    xtrain_t0 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_t0]
    xtrain_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t1]
    xtrain_t1 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_t1]
    xtrain_t0 = reduce(hcat, xtrain_t0)
    ω1 = [s[2] for s in xtrain_t1]
    ω2 = [s[4] for s in xtrain_t1]
    ytrain = [ω1, ω2]

    storage = Storage{Float64}(300, length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = _storage.x[id][1]
        storage.q[id][1] = _storage.q[id][1]
        storage.v[id][1] = _storage.v[id][1]
        storage.ω[id][1] = _storage.ω[id][1]
    end

    stdx = std(xtrain_t0, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (10 ./(stdx))...]

    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_t0, yi, MeanZero(), kernel)
        # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    minstate = max2mincoordinates(getcstate(_storage, 1), mechanism)
    θ1old, ω1old, θ2old, ω2old = minstate[1], minstate[2], minstate[1]+minstate[5], minstate[2]+minstate[6]
    minstate = max2mincoordinates(getcstate(_storage, 2), mechanism)
    θ1curr, θ2curr = minstate[1], minstate[1]+minstate[5]
    for i in 2:length(storage.x[1])
        ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
        θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
        storage.x[1][i] = [0, 0.5sin(θ1curr), -0.5cos(θ1curr)]
        storage.x[2][i] = [0, sin(θ1curr) + 0.5sin(θ2curr), -cos(θ1curr) - 0.5cos(θ2curr)]
        storage.x[3][i] = [0, 0.5sin(θ2curr), -0.5cos(θ2curr)]
        storage.x[4][i] = [0, sin(θ2curr) + 0.5sin(θ1curr), -cos(θ2curr) - 0.5cos(θ1curr)]
        storage.q[1][i] = UnitQuaternion(RotX(θ1curr))
        storage.q[2][i] = UnitQuaternion(RotX(θ2curr))
        storage.q[3][i] = UnitQuaternion(RotX(θ2curr))
        storage.q[4][i] = UnitQuaternion(RotX(θ1curr))

        θ1curr = θ1curr + ω1curr*mechanism.Δt  # ω1*Δt
        θ2curr = θ2curr + ω2curr*mechanism.Δt  # ω2*Δt

    end
    return _storage, mechanism
end
