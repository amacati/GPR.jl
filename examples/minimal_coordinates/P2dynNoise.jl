using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyP2Min(config)
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = [SMatrix{3,3,Float64}(Σ["J"]randn(9)...), SMatrix{3,3,Float64}(Σ["J"]randn(9)...)]
    m = abs.(ones(2) .+ Σ["m"]randn(2))
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = doublependulum2D(Δt=config["Δtsim"], θstart=[θ1, θ2], m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])
        dataset += storage
    end
    mechanism = doublependulum2D(Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l1, l2 = mechanism.bodies[1].shape.xyz[3], mechanism.bodies[2].shape.xyz[3]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]

    xtest_t1true, xtest_tktrue = deepcopy(sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, 
                                                        pseudorandom = true, exclude = trainsets, stepsahead=[1,config["simsteps"]+1]))
    xtest_t1true = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1true]  # Noise free

    # Add noise to the dataset
    for storage in dataset.storages
        for t in 1:length(storage.x[1])
            storage.q[1][t] = UnitQuaternion(RotX(Σ["q"]*randn())) * storage.q[1][t]
            storage.ω[1][t] += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            storage.q[2][t] = UnitQuaternion(RotX(Σ["q"]*randn())) * storage.q[2][t]
            storage.ω[2][t] += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ1 = Rotations.rotation_angle(storage.q[1][t])*sign(storage.q[1][t].x)*sign(storage.q[1][t].w)  # Signum for axis direction
            θ2 = Rotations.rotation_angle(storage.q[2][t])*sign(storage.q[2][t].x)*sign(storage.q[2][t].w) - θ1
            ω1, ω2 = storage.ω[1][t][1], storage.ω[2][t][1]
            storage.x[1][t] = [0, l1/2*sin(θ1), -l1/2*cos(θ1)]  # Noise is consequence of θ and ω
            storage.v[1][t] = [0, ω1*cos(θ1)*l1/2, ω1*sin(θ1)*l1/2]  # l/2 because measurement is in the center of the pendulum
            storage.x[2][t] = [0, l1*sin(θ1) + l2/2*sin(θ1+θ2), -l1*cos(θ1) - l2/2*cos(θ1+θ2)]  # Noise is consequence of θ and ω
            storage.v[2][t] = [0, l1*cos(θ1)*ω1 + l2/2*cos(θ1 + θ2)*(ω1 + ω2),
                               l1*sin(θ1) + l2/2*sin(θ1 + θ2)*(ω1 + ω2)]
        end
    end 
    # Create train and testsets
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    xtrain_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t0]
    xtrain_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t1]
    xtrain_t0 = reduce(hcat, xtrain_t0)
    yω1 = [s[2] for s in xtrain_t1]
    yω2 = [s[4] for s in xtrain_t1]
    ytrain = [yω1, yω2]
    xtest_t0 = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, pseudorandom = true, exclude = trainsets, stepsahead = [0])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]  # Noisy

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    params = config["params"]
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, id, 4, coords = "min")
        gp = GP(xtrain_t0, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_t0)
        θ1old, ω1old, θ2old, ω2old = xtest_t0[i]
        θ1curr, _, θ2curr, _ = xtest_t1true[i]
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
    return predictedstates, xtest_tktrue
end
