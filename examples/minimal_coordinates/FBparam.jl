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
    xtrain_old = [tocstate(x) for x in config["traindf"].sold]
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_old = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_old]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_curr = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    ω1 = [s[2] for s in xtrain_curr]
    ω2 = [s[4] for s in xtrain_curr]
    ytrain = [ω1, ω2]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_curr = [tocstate(x) for x in config["testdf"].scurr]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]
    xtest_old = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtest_old]
    xtest_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtest_curr]
    xtest_curr = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtest_curr]
    # intentionally not converting xtest_future since final comparison is done in maximal coordinates

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1., (10 ./(stdx))...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_old)
        θ1old, ω1old, θ2old, ω2old = xtest_old[i]
        θ1curr, _, θ2curr, _ = xtest_curr[i]
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
    return predictedstates, xtest_future, params
end

function simulation()
    dataset = Dataset()
    _storage, mechanism, initialstates = fourbar(θstart = [π/4, π/4])
    l = mechanism.bodies[1].shape.xyz[3]
    dataset += _storage
    xtrain_old, xtrain_curr = sampledataset(dataset, 200, random = true, replace=false)
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_old = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_old]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_curr = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    ω1 = [s[2] for s in xtrain_curr]
    ω2 = [s[4] for s in xtrain_curr]
    ytrain = [ω1, ω2]

    storage = Storage{Float64}(300, length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = _storage.x[id][1]
        storage.q[id][1] = _storage.q[id][1]
        storage.v[id][1] = _storage.v[id][1]
        storage.ω[id][1] = _storage.ω[id][1]
    end

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (10 ./(stdx))...]

    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
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
