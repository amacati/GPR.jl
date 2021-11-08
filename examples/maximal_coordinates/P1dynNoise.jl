using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyP1Max(config)
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs(1. + Σ["m"]randn())
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=config["Δtsim"], θstart=θ, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])
        dataset += storage
    end
    mechanism = simplependulum2D(Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.rh[2]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
    xtest_t0true, xtest_tktrue = deepcopy(sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true,
                                                        pseudorandom = true, exclude = trainsets, stepsahead=[0,config["simsteps"]+1]))

    # Add noise to the dataset
    for storage in dataset.storages
        for t in 1:length(storage.x[1])
            storage.q[1][t] = UnitQuaternion(RotX(Σ["q"]*randn())) * storage.q[1][t]  # Small error around θ
            storage.ω[1][t] += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ = Rotations.rotation_angle(storage.q[1][t])*sign(storage.q[1][t].x)*sign(storage.q[1][t].w)  # Signum for axis direction
            ω = storage.ω[1][t][1]
            storage.x[1][t] = [0, l/2*sin(θ), -l/2*cos(θ)]  # Noise is consequence of θ and ω
            storage.v[1][t] = [0, ω*cos(θ)*l/2, ω*sin(θ)*l/2]  # l/2 because measurement is in the center of the pendulum
        end
    end

    # Create train and testsets
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    xtrain_t0 = reduce(hcat, xtrain_t0)
    yv2 = [s[9] for s in xtrain_t1]
    yv3 = [s[10] for s in xtrain_t1]
    yω = [s[11] for s in xtrain_t1]
    y_train = [yv2, yv3, yω]
    xtest_t0 = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, pseudorandom = true, exclude = trainsets, stepsahead = [0])

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    for (id, yi) in enumerate(y_train)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, 1, id+1)  # [2, 3, 4] -> vy, vz, ωx
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
            μ = predict_velocities(gps, reshape(oldstates, :, 1))
            vcurr, ωcurr = [SVector(0, μ[1:2]...)], [SVector(μ[3], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_tktrue
end
