using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "utils.jl"))


function experimentCPMin(config, params)
    l = 0.5
    mechanism = deepcopy(config["mechanism"])
    predictedstates = Vector{Vector{Float64}}()
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

    for i in 1:length(config["x_test"])
        xold, vold, θold, ωold = config["x_test"][i]
        xcurr, _, θcurr, _ = config["xnext_test"][i]
        vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, θold, ωold], :, 1))
        xnew = xcurr + vcurr*mechanism.Δt
        θnew = θcurr + ωcurr*mechanism.Δt
        cstates = [0, xnew, zeros(12)..., 0.5l*sin(θnew)+xnew, -0.5l*cos(θnew), zeros(10)...]  # Only position matters for prediction error
        push!(predictedstates, cstates)
    end
    return predictedstates
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
    xcurr, _, θcurr = config["xnext_test"][1]
    for i in 2:length(storage.x[1])
        vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, θold, ωold], :, 1))
        storage.x[1][i] = [0, xcurr, 0]
        storage.q[1][i] = one(UnitQuaternion)
        storage.x[2][i] = [0, xcurr+0.5l*sin(θcurr), -0.5l*cos(θcurr)]
        storage.q[2][i] = UnitQuaternion(RotX(θcurr))
        xold, vold, θold, ωold = xcurr, vcurr, θcurr,ωcurr
        xcurr += vcurr*mechanism.Δt
        θcurr += ωcurr*mechanism.Δt
    end
    return storage
end

#storage = simulation(config, params)
#ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
