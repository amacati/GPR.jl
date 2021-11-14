using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics
using JSON


function experimentCPMin(config)
    mechanism = deepcopy(config["mechanism"])
    l = mechanism.bodies[2].shape.rh[2]
    # Sample from dataset
    xtrain_old = [tocstate(x) for x in config["traindf"].sold]
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    yv = [s[2] for s in xtrain_curr]
    yω = [s[4] for s in xtrain_curr]
    ytrain = [yv, yω]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_curr = [tocstate(x) for x in config["testdf"].scurr]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]
    xtest_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtest_curr]
    # intentionally not converting xtest_future since final comparison is done in maximal coordinates
    
    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (50 ./(stdx))...]
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
        xold, vold, θold, ωold = xtest_old[i]
        xcurr, _, θcurr, _ = xtest_curr[i]
        for _ in 1:config["simsteps"]
            vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, θold, ωold], :, 1))
            xold, vold, θold, ωold = xcurr, vcurr, θcurr, ωcurr
            xcurr = xcurr + vcurr*mechanism.Δt
            θcurr = θcurr + ωcurr*mechanism.Δt
        end
        q = UnitQuaternion(RotX(θcurr))
        vq = [q.w, q.x, q.y, q.z]
        cstate = [0, xcurr, 1, zeros(11)..., 0.5l*sin(θcurr)+xcurr, -0.5l*cos(θcurr), vq..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_future, params
end


function simulation(config, params)
    l = 0.5
    mechanism = deepcopy(config["mechanism"])
    storage = Storage{Float64}(300, length(mechanism.bodies))
    x, _, θ, _ = config["x_test"][1]
    storage.x[1][1] = [0, x, 0]
    storage.q[1][1] = one(UnitQuaternion)
    storage.x[2][1] = [0, x+0.5l*sin(θ), -0.5l*cos(θ)]
    storage.q[2][1] = UnitQuaternion(RotX(θ))

    gps = Vector()
    for yi in config["y_train"]
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config["x_train"], yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    xold, vold, θold, ωold = config["x_test"][1]
    xcurr, _, θcurr, _ = config["xnext_test"][1]
    for i in 2:length(storage.x[1])
        vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, θold, ωold], :, 1))
        storage.x[1][i] = [0, xcurr, 0]
        storage.q[1][i] = one(UnitQuaternion)
        storage.x[2][i] = [0, xcurr+0.5l*sin(θcurr), -0.5l*cos(θcurr)]
        storage.q[2][i] = UnitQuaternion(RotX(θcurr))
        xold, vold, θold, ωold = xcurr, vcurr, θcurr, ωcurr
        xcurr += vcurr*mechanism.Δt
        θcurr += ωcurr*mechanism.Δt
    end
    return storage
end
