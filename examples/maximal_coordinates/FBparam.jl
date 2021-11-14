using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentFBMax(config)
    mechanism = deepcopy(config["mechanism"])
    # Sample from dataset
    xtrain_old = [tocstate(x) for x in config["traindf"].sold]
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]
    xtrain_old = reduce(hcat, xtrain_old)
    yv12 = [s[9] for s in xtrain_curr]
    yv13 = [s[10] for s in xtrain_curr]
    yv22 = [s[22] for s in xtrain_curr]
    yv23 = [s[23] for s in xtrain_curr]
    yv32 = [s[35] for s in xtrain_curr]
    yv33 = [s[36] for s in xtrain_curr]
    yv42 = [s[48] for s in xtrain_curr]
    yv43 = [s[49] for s in xtrain_curr]
    yω11 = [s[11] for s in xtrain_curr]
    yω21 = [s[24] for s in xtrain_curr]
    yω31 = [s[37] for s in xtrain_curr]
    yω41 = [s[50] for s in xtrain_curr]
    ytrain = [yv12, yv13, yv22, yv23, yv32, yv33, yv42, yv43, yω11, yω21, yω31, yω41]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]

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
        setstates!(mechanism, tovstate(xtest_old[i]))
        oldstates = xtest_old[i]
        for _ in 1:config["simsteps"]
            μ = predict_velocities(gps, reshape(reduce(vcat, oldstates), :, 1))
            vcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...), SVector(0, μ[5:6]...), SVector(0, μ[7:8]...)]
            ωcurr = [SVector(μ[9], 0, 0), SVector(μ[10], 0, 0), SVector(μ[11], 0, 0), SVector(μ[12], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism, regularizer = 1e-10)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_future, params
end

function simulation()
    dataset = Dataset()
    _storage, mechanism, initialstates = fourbar(θstart = [π/4, π/8])
    l = mechanism.bodies[1].shape.xyz[3]
    dataset += _storage
    xtrain_old, xtrain_curr = sampledataset(dataset, 100, random = true, replace=false)
    xtrain_old = reduce(hcat, xtrain_old)
    yv12 = [s[9] for s in xtrain_curr]
    yv13 = [s[10] for s in xtrain_curr]
    yv22 = [s[22] for s in xtrain_curr]
    yv23 = [s[23] for s in xtrain_curr]
    yv32 = [s[35] for s in xtrain_curr]
    yv33 = [s[36] for s in xtrain_curr]
    yv42 = [s[48] for s in xtrain_curr]
    yv43 = [s[49] for s in xtrain_curr]
    yω11 = [s[11] for s in xtrain_curr]
    yω21 = [s[24] for s in xtrain_curr]
    yω31 = [s[37] for s in xtrain_curr]
    yω41 = [s[50] for s in xtrain_curr]
    ytrain = [yv12, yv13, yv22, yv23, yv32, yv33, yv42, yv43, yω11, yω21, yω31, yω41]

    storage = Storage{Float64}(300, length(mechanism.bodies))
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = _storage.x[id][1]
        storage.q[id][1] = _storage.q[id][1]
        storage.v[id][1] = _storage.v[id][1]
        storage.ω[id][1] = _storage.ω[id][1]
    end

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1., (10 ./(stdx))...]

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

    setstates!(mechanism, getvstates(_storage, 1))
    states = getcstate(_storage, 1)
    for i in 2:length(storage.x[1])
        μ = predict_velocities(gps, reshape(states, :, 1))
        vcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...), SVector(0, μ[5:6]...), SVector(0, μ[7:8]...)]
        ωcurr = [SVector(μ[9], 0, 0), SVector(μ[10], 0, 0), SVector(μ[11], 0, 0), SVector(μ[12], 0, 0)]
        projectv!(vcurr, ωcurr, mechanism, regularizer = 1e-10)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getcstate(mechanism)
        overwritestorage(storage, tovstate(states), i)
    end
    return storage, mechanism
end
