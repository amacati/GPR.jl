using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "dataset.jl"))


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
    ytrain = [[s[2] for s in xtrain_t1]]
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
    gp = GP(xtrain_t0, ytrain[1], MeanZero(), kernel)
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

function experimentNoisyP1Min(config)
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1. .+ Σ["m"]randn())
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=Δtsim, θstart=θ, m = m, ΔJ = ΔJ)
        dataset += storage
    end
    mechanism = simplependulum2D(Δt=0.01, m = m, ΔJ = ΔJ)[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
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
    y_train = [s[2] for s in xnext_train]
    x_test, _, _ = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    x_test = [max2mincoordinates(cstate, mechanism) for cstate in x_test]  # Noisy

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(x_train, y_train, MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
    
    for i in 1:length(x_test)
        θold, ωold = x_test[i]  # Noisy
        θcurr_gt, _ = xnext_test_gt[i]  # Noise free
        ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
        θnew = θcurr_gt + ωcurr*mechanism.Δt  # ω1*Δt
        q1 = UnitQuaternion(RotX(θnew))
        vq1 = [q1.w, q1.x, q1.y, q1.z]
        cstate = [0, 0.5sin(θnew), -0.5cos(θnew), vq1..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates, xresult_test_gt
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
