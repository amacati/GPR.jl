using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentP2Min(config)
    mechanism = deepcopy(config["mechanism"])
    l1, l2 = mechanism.bodies[1].shape.xyz[3], mechanism.bodies[2].shape.xyz[3]
    # Sample from dataset
    xtrain_old = [tocstate(x) for x in config["traindf"].sold]
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    yω1 = [s[2] for s in xtrain_curr]
    yω2 = [s[4] for s in xtrain_curr]
    ytrain = [yω1, yω2]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_curr = [tocstate(x) for x in config["testdf"].scurr]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]
    xtest_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtest_curr]
    # intentionally not converting xtest_future since final comparison is done in maximal coordinates

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1.1, (50 ./stdx)...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
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
        q1, q2 = UnitQuaternion(RotX(θ1curr)), UnitQuaternion(RotX(θ1curr + θ2curr))
        vq1, vq2 = [q1.w, q1.x, q1.y, q1.z], [q2.w, q2.x, q2.y, q2.z]
        cstate = [0, 0.5l1*sin(θ1curr), -0.5l1*cos(θ1curr), vq1..., zeros(6)...,
                   0, l1*sin(θ1curr) + 0.5l2*sin(θ1curr+θ2curr), -l1*cos(θ1curr) - 0.5l2*cos(θ1curr + θ2curr), vq2..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_future, params
end

function simulation(config, params)
    l2 = sqrt(2) / 2  # Length param of the second pendulum link
    mechanism = deepcopy(config["mechanism"])
    storage = Storage{Float64}(300, length(mechanism.bodies))
    θ1, _, θ2, _ = config["x_test"][1]
    storage.x[1][1] = [0, 0.5sin(θ1), -0.5cos(θ1)]
    storage.q[1][1] = UnitQuaternion(RotX(θ1))
    storage.x[2][1] = [0, sin(θ1) + 0.5l2*sin(θ1+θ2), -cos(θ1) - 0.5l2*cos(θ1 + θ2)]
    storage.q[2][1] = UnitQuaternion(RotX(θ1 + θ2))

    gps = Vector()
    for yi in config["y_train"]
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config["x_train"], yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    θ1old, ω1old, θ2old, ω2old = config["x_test"][1]
    θ1curr, _, θ2curr, _ = config["xnext_test"][1]
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
