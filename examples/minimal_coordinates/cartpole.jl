using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics
using JSON

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

function experimentNoisyCPMin(config, params)
    l = 0.5
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = [SMatrix{3,3,Float64}(Σ["J"]randn(9)...), SMatrix{3,3,Float64}(Σ["J"]randn(9)...)]
    m = abs.(ones(2) .+ Σ["m"]randn(2))
    for θstart in -π:1:π, vstart in -1:1:1, ωstart in -1:1:1
        storage, _, _ = cartpole(Δt=Δtsim, θstart=θstart, vstart=vstart, ωstart=ωstart, m = m, ΔJ = ΔJ)
        dataset += storage
    end
    mechanism = cartpole(Δt=0.01, m = m, ΔJ = ΔJ)[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    x_test_gt, xnext_test_gt, xresult_test_gt = deepcopy(sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)]))
    x_test_gt = [max2mincoordinates(cstate, mechanism) for cstate in x_test_gt]  # Noise free
    xnext_test_gt = [max2mincoordinates(cstate, mechanism) for cstate in xnext_test_gt]  # Noise free

    # Add noise to the dataset
    for storage in dataset.storages
        for id in 1:length(storage.x)
            for t in 1:length(storage.x[id])
                storage.x[id][t] += Σ["x"]*[0, randn(2)...]  # Pos noise, x is fixed
                storage.q[id][t] = UnitQuaternion(RotX(Σ["q"]*randn())) * storage.q[id][t]
                storage.v[id][t] += Σ["v"]*[0, randn(2)...]  # Zero noise in fixed vx
                storage.ω[id][t] += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            end
        end
    end

    # Create train and testsets
    x_train, xnext_train, _ = sampledataset(dataset, config["nsamples"], Δt = Δtsim, exclude = testsets)
    x_train = [max2mincoordinates(cstate, mechanism) for cstate in x_train]
    xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]
    x_train = reduce(hcat, x_train)
    yv = [s[2] for s in xnext_train]
    yω = [s[4] for s in xnext_train]
    y_train = [yv, yω]
    x_test, _, _ = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    x_test = [max2mincoordinates(cstate, mechanism) for cstate in x_test]  # Noisy

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in y_train
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(x_train, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end
    
    for i in 1:length(x_test)
        xold, vold, θold, ωold = x_test[i]  # Noisy
        xcurr, _, θcurr, _ = xnext_test_gt[i]  # Noise free
        vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, θold, ωold], :, 1))
        xnew = xcurr + vcurr*mechanism.Δt
        θnew = θcurr + ωcurr*mechanism.Δt
        cstate = [0, xnew, zeros(12)..., 0.5l*sin(θnew)+xnew, -0.5l*cos(θnew), zeros(10)...]  # Only position matters for prediction error
        push!(predictedstates, cstate)
    end
    return predictedstates, xresult_test_gt
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
        # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
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

#=
function get_config()
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    nsamples = 512
    n = 1
    for θstart in -π:1:π, vstart in -1:1:1, ωstart in -1:1:1
        ωstart == -2 && display((ωstart,n))
        n += 1
        storage, _, _ = cartpole(Δt=Δtsim, θstart=θstart, vstart=vstart, ωstart=ωstart)
        dataset += storage
    end
    mechanism = cartpole(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    display(testsets)
    x_train, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    x_train = [max2mincoordinates(cstate, mechanism) for cstate in x_train]
    xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]
    x_train = reduce(hcat, x_train)
    yv = [s[2] for s in xnext_train]
    yω = [s[4] for s in xnext_train]
    y_train = [yv, yω]
    x_test, xnext_test, _ = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    x_test = [max2mincoordinates(cstate, mechanism) for cstate in x_test]
    xnext_test = [max2mincoordinates(cstate, mechanism) for cstate in xnext_test]
    config = Dict("mechanism"=>mechanism, 
                  "x_train"=>x_train, 
                  "y_train"=>y_train,
                  "x_test"=>x_test,
                  "xnext_test"=>xnext_test,
                   )
    return config
end

function loadconfig()
    open(joinpath(dirname(dirname(@__FILE__)), "config", "config.json"),"r") do f
        return JSON.parse(f)
    end
end

cnfg = loadconfig()
storage = simulation(get_config(), cnfg["CP_MIN512_FINAL"])
ConstrainedDynamicsVis.visualize(config["mechanism"], storage; showframes = true, env = "editor")
=#