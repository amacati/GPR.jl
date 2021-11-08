using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentP1Min(config)
    mechanism = deepcopy(config["mechanism"])
    l = mechanism.bodies[1].shape.rh[2]
    # Sample from dataset
    dataset = config["dataset"]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    
    xtrain_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t0]
    xtrain_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_t1]
    xtrain_t0 = reduce(hcat, xtrain_t0)
    ytrain = [s[2] for s in xtrain_t1]
    xtest_t0, xtest_t1, xtest_tk = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, 
                                                 pseudorandom = true, exclude = trainsets, stepsahead=[0,1,config["simsteps"]+1])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]
    xtest_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1]
    # intentionally not converting xtest_tk since final comparison is done in maximal coordinates

    stdx = std(xtrain_t0, dims=2)
    stdx[stdx .== 0] .= 100
    params = [1.1, (10 ./(stdx))...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(xtrain_t0, ytrain, MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))

    for i in 1:length(xtest_t0)
        θold, ωold = xtest_t0[i]
        θcurr, _ = xtest_t1[i]
        for _ in 1:config["simsteps"]
            ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
            θold, ωold = θcurr, ωcurr
            θcurr = θcurr + ωcurr*mechanism.Δt
        end
        q1 = UnitQuaternion(RotX(θcurr))
        vq1 = [q1.w, q1.x, q1.y, q1.z]
        cstate = [0, 0.5l*sin(θcurr), -0.5l*cos(θcurr), vq1..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_tk, params
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
