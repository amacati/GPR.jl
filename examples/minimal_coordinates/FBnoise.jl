using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentNoisyFBMin(config)
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1 .+ Σ["m"]randn())
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = fourbar(Δt=config["Δtsim"], θstart=[θ1, θ2], m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])
        dataset += storage
    end
    mechanism = fourbar(Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.xyz[3]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
    xtest_t1true, xtest_tktrue = deepcopy(sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true,
                                                        pseudorandom = true, exclude = trainsets, stepsahead=[1,config["simsteps"]+1]))
    xtest_t1true = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1true]
    xtest_t1true = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtest_t1true]

    # Add noise to the dataset
    for storage in dataset.storages
        for t in 1:length(storage.x[1])
            storage.q[1][t] = UnitQuaternion(RotX(Σ["q"]*randn())) * storage.q[1][t]
            storage.ω[1][t] += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            storage.q[3][t] = UnitQuaternion(RotX(Σ["q"]*randn())) * storage.q[3][t]
            storage.ω[3][t] += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ1 = Rotations.rotation_angle(storage.q[1][t])*sign(storage.q[1][t].x)*sign(storage.q[1][t].w)  # Signum for axis direction
            θ2 = Rotations.rotation_angle(storage.q[3][t])*sign(storage.q[3][t].x)*sign(storage.q[3][t].w)
            ω1, ω2 = storage.ω[1][t][1], storage.ω[3][t][1]
            storage.q[2][t] = UnitQuaternion(RotX(θ2))
            storage.q[4][t] = UnitQuaternion(RotX(θ1))
            storage.x[1][t] = [0, 0.5sin(θ1)l, -0.5cos(θ1)l]
            storage.x[2][t] = [0, sin(θ1)l + 0.5sin(θ2)l, -cos(θ1)l - 0.5cos(θ2)l]
            storage.x[3][t] = [0, 0.5sin(θ2)l, -0.5cos(θ2)l]
            storage.x[4][t] = [0, sin(θ2)l + 0.5sin(θ1)l, -cos(θ2)l - 0.5cos(θ1)l]
            storage.v[1][t] = [0, 0.5cos(θ1)l*ω1, 0.5sin(θ1)l*ω1]
            storage.v[2][t] = [0, cos(θ1)l*ω1 + 0.5cos(θ2)l*ω2, sin(θ1)l*ω1 + 0.5sin(θ2)l*ω2]
            storage.v[3][t] = [0, 0.5cos(θ2)l*ω2, 0.5sin(θ2)l*ω2]
            storage.v[4][t] = [0, cos(θ2)l*ω2 + 0.5cos(θ1)l*ω1, sin(θ2)l*ω2 + 0.5sin(θ1)l*ω1]
            storage.ω[2][t] = storage.ω[2][t]
            storage.ω[4][t] = storage.ω[1][t]
        end
    end
    # Create train and testsets
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    xtrain_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t0]
    xtrain_t0 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_t0]
    xtrain_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t1]
    xtrain_t1 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtrain_t1]
    xtrain_t0 = reduce(hcat, xtrain_t0)
    ω1 = [s[2] for s in xtrain_t1]
    ω2 = [s[4] for s in xtrain_t1]
    ytrain = [ω1, ω2]
    xtest_t0 = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, pseudorandom = true, exclude = trainsets, stepsahead = [0])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]
    xtest_t0 = [[x[1:2]..., x[1]+x[5], x[2]+x[6]] for x in xtest_t0]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_t0, yi, MeanZero(), kernel)
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
        x1 = [0, 0.5sin(θ1curr), -0.5cos(θ1curr)]
        x2 = [0, sin(θ1curr) + 0.5sin(θ2curr), -cos(θ1curr) - 0.5cos(θ2curr)]
        x3 = [0, 0.5sin(θ2curr), -0.5cos(θ2curr)]
        x4 = [0, sin(θ2curr) + 0.5sin(θ1curr), -cos(θ2curr) - 0.5cos(θ1curr)]
        cstate = [x1..., zeros(10)..., x2..., zeros(10)..., x3..., zeros(10)..., x4..., zeros(10)...]  # Orientation, velocities not used in error
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_tktrue
end