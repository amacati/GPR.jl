using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyCPMax(config)
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = [SMatrix{3,3,Float64}(Σ["J"]randn(9)...), SMatrix{3,3,Float64}(Σ["J"]randn(9)...)]
    m = abs.(ones(2) .+ Σ["m"]randn(2))
    for θstart in -π:1:π, vstart in -1:1:1, ωstart in -1:1:1
        storage, _, _ = cartpole(Δt=config["Δtsim"], θstart=θstart, vstart=vstart, ωstart=ωstart,
                                 m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])
        dataset += storage
    end
    mechanism = cartpole(Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    l = mechanism.bodies[2].shape.rh[2]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
    xtest_t0true, xtest_tktrue = deepcopy(sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true,
                                                        pseudorandom = true, exclude = trainsets, stepsahead=[0,config["simsteps"]+1]))
    # Add noise to the dataset
    for storage in dataset.storages
        for t in 1:length(storage.x[1])
            storage.x[1][t] += Σ["x"]*[0, randn(), 0]  # Cart pos noise only y, no orientation noise
            storage.v[1][t] += Σ["v"]*[0, randn(), 0]  # Same for v
            storage.q[2][t] = UnitQuaternion(RotX(Σ["q"]*randn())) * storage.q[2][t]
            storage.ω[2][t] += Σ["ω"]*[randn(), 0, 0]
            θ = Rotations.rotation_angle(storage.q[2][t])*sign(storage.q[2][t].x)*sign(storage.q[2][t].w)  # Signum for axis direction
            ω = storage.ω[2][t][1]
            storage.x[2][t] = storage.x[1][t] + [0, l/2*sin(θ), -l/2*cos(θ)]
            storage.v[2][t] = storage.v[1][t] + [0, ω*cos(θ)*l/2, ω*sin(θ)*l/2]
        end
    end
    # Create train and testsets
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    xtrain_t0 = reduce(hcat, xtrain_t0)
    yv12 = [s[9] for s in xtrain_t1]
    yv22 = [s[22] for s in xtrain_t1]
    yv23 = [s[23] for s in xtrain_t1]
    yω21 = [s[24] for s in xtrain_t1]
    ytrain = [yv12, yv22, yv23, yω21]
    
    xtest_t0 = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, pseudorandom = true, exclude = trainsets, stepsahead = [0])

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        id == 1 ? bodyID = 1 : bodyID = 2
        id in [1, 2] ? entryID = 2 : entryID = entryID = id
        mean = MeanDynamics(mechanism, bodyID, entryID)
        gp = GP(xtrain_t0, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_t0)
        setstates!(mechanism, tovstate(xtest_t0true[i]))
        oldstates = xtest_t0[i]
        for _ in 1:config["simsteps"]
            μ = predict_velocities(gps, reshape(xtest_t0[i], :, 1))
            vcurr, ωcurr = [SVector(0, μ[1], 0), SVector(0, μ[2:3]...)], [SVector(zeros(3)...), SVector(μ[4], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_tktrue
end
