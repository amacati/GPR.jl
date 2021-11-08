using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentNoisyFBMax(config)
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
    xtest_t0true, xtest_tktrue = deepcopy(sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true,
                                                        pseudorandom = true, exclude = trainsets, stepsahead=[0,config["simsteps"]+1]))
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
    xtrain_t0 = reduce(hcat, xtrain_t0)
    yv12 = [s[9] for s in xtrain_t1]
    yv13 = [s[10] for s in xtrain_t1]
    yv22 = [s[22] for s in xtrain_t1]
    yv23 = [s[23] for s in xtrain_t1]
    yv32 = [s[35] for s in xtrain_t1]
    yv33 = [s[36] for s in xtrain_t1]
    yv42 = [s[48] for s in xtrain_t1]
    yv43 = [s[49] for s in xtrain_t1]
    yω11 = [s[11] for s in xtrain_t1]
    yω21 = [s[24] for s in xtrain_t1]
    yω31 = [s[37] for s in xtrain_t1]
    yω41 = [s[50] for s in xtrain_t1]
    ytrain = [yv12, yv13, yv22, yv23, yv32, yv33, yv42, yv43, yω11, yω21, yω31, yω41]
    xtest_t0 = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, pseudorandom = true, exclude = trainsets, stepsahead = [0])

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
        setstates!(mechanism, tovstate(xtest_t0true[i]))
        oldstates = xtest_t0[i]
        for _ in 1:config["simsteps"]
            μ = predict_velocities(gps, reshape(oldstates, :, 1))  # Noisy
            vcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...), SVector(0, μ[5:6]...), SVector(0, μ[7:8]...)]
            ωcurr = [SVector(μ[9], 0, 0), SVector(μ[10], 0, 0), SVector(μ[11], 0, 0), SVector(μ[12], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_tktrue
end
