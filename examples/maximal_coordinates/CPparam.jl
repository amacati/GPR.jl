using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentCPMax(config)
    mechanism = deepcopy(config["mechanism"])
    # Sample from dataset
    xtrain_old = [tocstate(x) for x in config["traindf"].sold]
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]
    xtrain_old = reduce(hcat, xtrain_old)
    yv12 = [s[9] for s in xtrain_curr]
    yv22 = [s[22] for s in xtrain_curr]
    yv23 = [s[23] for s in xtrain_curr]
    yω21 = [s[24] for s in xtrain_curr]
    ytrain = [yv12, yv22, yv23, yω21]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]

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
        setstates!(mechanism, tovstate(xtest_old[i]))
        oldstates = xtest_old[i]
        for _ in 1:config["simsteps"]
            μ = predict_velocities(gps, reshape(reduce(vcat, oldstates), :, 1))
            vcurr, ωcurr = [SVector(0, μ[1], 0), SVector(0, μ[2:3]...)], [SVector(zeros(3)...), SVector(μ[4], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_future, params
end

function simulation(config, params)
    mechanism = deepcopy(config["mechanism"])
    storage = Storage{Float64}(300, length(mechanism.bodies))
    initialstates = tovstate(config["x_test"][1])
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = initialstates[id][1:3]
        storage.q[id][1] = UnitQuaternion(initialstates[id][4], initialstates[id][5:7])
        storage.v[id][1] = initialstates[id][8:10]
        storage.ω[id][1] = initialstates[id][11:13]
    end

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

    states = tovstate(config["x_test"][1])
    setstates!(mechanism, states)
    for i in 2:length(storage.x[1])
        μ = predict_velocities(gps, reshape(reduce(vcat, states), :, 1))
        vcurr, ωcurr = [SVector(0, μ[1], 0), SVector(0, μ[2:3]...)], [SVector(zeros(3)...), SVector(μ[4], 0, 0)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getvstates(mechanism)
        overwritestorage(storage, states, i)
    end
    return storage
end