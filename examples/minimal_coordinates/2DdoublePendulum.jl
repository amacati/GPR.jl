using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "utils.jl"))
# TODO: REMOVE
include(joinpath("..", "dataset.jl"))
include(joinpath("..", "generatedata.jl"))


function experimentP2Min(config)
    mechanism = deepcopy(config["mechanism"])
    l1, l2 = mechanism.bodies[1].shape.xyz[3], mechanism.bodies[2].shape.xyz[3]
    # Sample from dataset
    dataset = config["dataset"]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    xtrain_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t0]
    xtrain_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t1]
    xtrain_t0 = reduce(hcat, xtrain_t0)
    yω1 = [s[2] for s in xtrain_t1]
    yω2 = [s[4] for s in xtrain_t1]
    ytrain = [yω1, yω2]
    xtest_t0, xtest_t1, xtest_tk = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, 
                                                 pseudorandom = true, exclude = trainsets, stepsahead=[0,1,config["simsteps"]+1])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]
    xtest_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1]
    # intentionally not converting xtest_tk since final comparison is done in maximal coordinates

    stdx = std(xtrain_t0, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1.1, (50 ./stdx)...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
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
        θ1curr, _, θ2curr, _ = xtest_t1[i]
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
    return predictedstates, xtest_tk, params
end

function experimentNoisyP2Min(config)
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = [SMatrix{3,3,Float64}(Σ["J"]randn(9)...), SMatrix{3,3,Float64}(Σ["J"]randn(9)...)]
    m = abs.(ones(2) .+ Σ["m"]randn(2))
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = doublependulum2D(Δt=Δtsim, θstart=[θ1, θ2], m = m, ΔJ = ΔJ)
        dataset += storage
    end
    mechanism = doublependulum2D(Δt=0.01, m = m, ΔJ = ΔJ)[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
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
    yω1 = [s[2] for s in xnext_train]
    yω2 = [s[4] for s in xnext_train]
    y_train = [yω1, yω2]
    x_test, _, _ = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    x_test = [max2mincoordinates(cstate, mechanism) for cstate in x_test]  # Noisy

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    params = config["params"]
    for yi in y_train
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(x_train, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end
    l2 = sqrt(2) / 2
    for i in 1:length(x_test)
        θ1old, ω1old, θ2old, ω2old = x_test[i]  # Noisy
        θ1curr_gt, _, θ2curr_gt, _ = xnext_test_gt[i]  # Noise free
        ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
        θ1new = θ1curr_gt + ω1curr*mechanism.Δt  # ω1*Δt
        θ2new = θ2curr_gt + ω2curr*mechanism.Δt  # ω2*Δt
        q1, q2 = UnitQuaternion(RotX(θ1new)), UnitQuaternion(RotX(θ1new + θ2new))
        vq1, vq2 = [q1.w, q1.x, q1.y, q1.z], [q2.w, q2.x, q2.y, q2.z]
        cstates = [0, 0.5sin(θ1new), -0.5cos(θ1new), vq1..., zeros(6)...,
                   0, sin(θ1new) + 0.5l2*sin(θ1new+θ2new), -cos(θ1new) - 0.5l2*cos(θ1new + θ2new), vq2..., zeros(6)...]
        push!(predictedstates, cstates)    
    end
    return predictedstates, xresult_test_gt
end

function simulation(config, params)
    # params = [1.1762814437505233, 29.533410088995723, 75.16325297865855, 24.231256821154147, 14.428238673265541]
    l2 = sqrt(2) / 2  # Length param of the second pendulum link
    mechanism = deepcopy(config["mechanism"])
    storage = Storage{Float64}(300, length(mechanism.bodies))
    θ1, _, θ2, _ = config["x_test"][1]
    storage.x[1][1] = [0, 0.5sin(θ1), -0.5cos(θ1)]
    storage.q[1][1] = UnitQuaternion(RotX(θ1))
    storage.x[2][1] = [0, sin(θ1) + 0.5l2*sin(θ1+θ2), -cos(θ1) - 0.5l2*cos(θ1 + θ2)]
    storage.q[2][1] = UnitQuaternion(RotX(θ1 + θ2))

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

    θ1old, ω1old, θ2old, ω2old = config["x_test"][1]
    θ1curr, _, θ2curr, _ = config["xnext_test"][1]
    for i in 2:length(storage.x[1])
        ω1curr, ω2curr = predict_velocities(gps, reshape([θ1old, ω1old, θ2old, ω2old], :, 1))
        storage.x[1][i] = [0, 0.5sin(θ1curr), -0.5cos(θ1curr)]
        storage.q[1][i] = UnitQuaternion(RotX(θ1curr))
        storage.x[2][i] = [0, sin(θ1curr) + 0.5l2*sin(θ1curr+θ2curr), -cos(θ1curr) - 0.5l2*cos(θ1curr + θ2curr)]
        storage.q[2][i] = UnitQuaternion(RotX(θ1curr + θ2curr))
        θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
        θ1curr = θ1curr + ω1curr*mechanism.Δt  # ω1*Δt
        θ2curr = θ2curr + ω2curr*mechanism.Δt  # ω2*Δt
    end
    return storage
end
