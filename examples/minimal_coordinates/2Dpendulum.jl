using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "parallelsearch.jl"))
include(joinpath("..", "dataset.jl"))


function experimentP1Min(config, params)
    mechanism = deepcopy(config["mechanism"])
    predictedstates = Vector{Vector{Float64}}()
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(config["x_train"], config["y_train"][1], MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))

    for i in 1:length(config["x_test"])
        θold, ωold = config["x_test"][i]
        θcurr, _ = config["xnext_test"][i]
        ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
        θnew = θcurr + ωcurr*mechanism.Δt
        q1 = UnitQuaternion(RotX(θnew))
        vq1 = [q1.w, q1.x, q1.y, q1.z]
        cstate = [0, 0.5sin(θnew), -0.5cos(θnew), vq1..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates
end

function simulation(config, params)
    mechanism = deepcopy(config["mechanism"])
    storage = Storage{Float64}(300, length(mechanism.bodies))
    θ = config["x_test"][1][1]
    storage.x[1][1] = [0, 0.5sin(θ), -0.5cos(θ)]
    storage.q[1][1] = UnitQuaternion(RotX(θ))

    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(config["x_train"], config["y_train"][1], MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))

    θold, ωold = config["x_test"][1]
    θcurr, _ = config["xnext_test"][1]
    for i in 2:length(storage.x[1])
        ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
        storage.x[1][i] = [0, 0.5sin(θcurr), -0.5cos(θcurr)]
        storage.q[1][i] = UnitQuaternion(RotX(θcurr))
        θold, ωold = θcurr, ωcurr
        θcurr = θcurr + ωcurr*mechanism.Δt  # ω*Δt
    end
    return storage
end

# storage = simulation(config, params)
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
