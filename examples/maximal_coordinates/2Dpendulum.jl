using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "utils.jl"))


function experimentP1Max(config, params)
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
        oldstates = tovstate(config["x_test"][i])
        setstates!(mechanism, oldstates)
        μ = predict_velocities(gps, reshape(reduce(vcat, oldstates), :, 1))
        vcurr, ωcurr = [SVector(0, μ[1:2]...)], [SVector(μ[3], 0, 0)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates
end

function experimentNoisyP1Max(config, params)
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs(1. + Σ["m"]randn())
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=Δtsim, θstart=θ, m = m, ΔJ = ΔJ)
        dataset += storage
    end
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    x_test_gt, _, xresult_test_gt = deepcopy(sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)]))

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
    x_train = reduce(hcat, x_train)
    yv2 = [s[9] for s in xnext_train]
    yv3 = [s[10] for s in xnext_train]
    yω = [s[11] for s in xnext_train]
    y_train = [yv2, yv3, yω]
    x_test, _, _ = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    mechanism = simplependulum2D(Δt=0.01, m = m, ΔJ = ΔJ)[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M

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
        oldstates = tovstate(x_test_gt[i])
        setstates!(mechanism, oldstates)
        μ = predict_velocities(gps, reshape(x_test[i], :, 1))
        vcurr, ωcurr = [SVector(0, μ[1:2]...)], [SVector(μ[3], 0, 0)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xresult_test_gt
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
        vcurr, ωcurr = [SVector(μ[1:3]...)], [SVector(μ[4:6]...)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getvstates(mechanism)
        overwritestorage(storage, states, i)
    end
    return storage
end

# storage = simulation(config, params)
# storage, mechanism, _ = simplependulum2D(Δt=0.01, θstart=collect(-π/2:0.1:π/2)[3])
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
