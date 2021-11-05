include("generatedata.jl")
include("utils.jl")
include("dataset.jl")
include("maximal_coordinates/2Dpendulum.jl")
include("maximal_coordinates/2DdoublePendulum.jl")
include("maximal_coordinates/cartpole.jl")
include("minimal_coordinates/2Dpendulum.jl")
include("minimal_coordinates/2DdoublePendulum.jl")
include("minimal_coordinates/cartpole.jl")
include("parallel.jl")


function loadcheckpoint_or_defaults(_loadcheckpoint)
    nprocessed = 0
    kstep_mse = []
    params = []
    if _loadcheckpoint
        checkpointdict = loadcheckpoint(EXPERIMENT_ID)
        nprocessed = checkpointdict["nprocessed"]
        kstep_mse = checkpointdict["kstep_mse"]
        params = checkpointdict["params"]
    end
    return nprocessed, kstep_mse, params
end

function get_config(EXPERIMENT_ID, nruns, mechanism, Δtsim, dataset, nsamples, ntestsets, testsamples, simsteps,
                    nprocessed, kstep_mse, params)
    config = Dict("EXPERIMENT_ID"=>EXPERIMENT_ID,
                  "nruns"=>nruns,
                  "mechanism"=>mechanism,
                  "Δtsim"=>Δtsim,
                  "dataset"=>dataset, 
                  "nsamples"=>nsamples,
                  "ntestsets"=>ntestsets,
                  "testsamples"=>testsamples,
                  "simsteps"=>simsteps, 
                  "nprocessed"=>nprocessed,
                  "kstep_mse"=>kstep_mse, 
                  "params"=>params,
                  "paramlock"=>ReentrantLock(), 
                  "checkpointlock"=>ReentrantLock(), 
                  "resultlock"=>ReentrantLock())
    return config
end
  
function hyperparametersearchP1Max(settings, nsamples, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P1_MAX"*string(nsamples)
    # Generate Dataset
    dataset = Dataset()
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=settings["Δtsim"], θstart=θ)
        dataset += storage
    end
    mechanism = simplependulum2D(Δt=0.01, θstart=0.)[2]  # Reset Δt to 0.01 in mechanism
    nprocessed, kstep_mse, params = loadcheckpoint_or_defaults(_loadcheckpoint)
    # Create config dictionary
    config = get_config(EXPERIMENT_ID, settings["nruns"], mechanism, settings["Δtsim"], dataset, nsamples,
                        settings["ntestsets"], settings["testsamples"], settings["simsteps"], nprocessed, kstep_mse, params)
    # Launch parallel search
    parallelsearch(experimentP1Max, config)
end

function hyperparametersearchP2Max(ntrials, nsamples, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P2_MAX"*string(nsamples)
    
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = doublependulum2D(Δt=Δtsim, θstart=[θ1, θ2])
        dataset += storage
    end
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    xtrain, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    xtrain = reduce(hcat, xtrain)
    yv12 = [s[9] for s in xnext_train]
    yv13 = [s[10] for s in xnext_train]
    yv22 = [s[22] for s in xnext_train]
    yv23 = [s[23] for s in xnext_train]
    yω11 = [s[11] for s in xnext_train]
    yω21 = [s[24] for s in xnext_train]
    ytrain = [yv12, yv13, yv22, yv23, yω11, yω21]
    xtest_t0, _, xtest_tk = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    mechanism = doublependulum2D(Δt=0.01, θstart=[0, 0])[2]  # Reset Δt to 0.01 in mechanism
    
    # Load checkpoints
    nprocessed, kstep_mse, params = get_hpsearch_params(_loadcheckpoint)

    # Generate random parameters
    stdx = std(xtrain, dims=2)
    stdx[stdx .== 0] .= 100
    params = [1.1, (1 ./(0.02 .*stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 0.999) .* params for _ in 1:ntrials-1]
    push!(paramtuples, params)  # Make sure initial params are also included

    # Create config dictionary
    config = get_config(EXPERIMENT_ID, mechanism, xtrain, ytrain, xtest_t0, [], xtest_tk, paramtuples,
                        nprocessed, kstep_mse, params)

    # Launch parallel search
    parallelsearch(experimentP2Max, config)
end

function hyperparametersearchCPMax(ntrials, nsamples, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "CP_MAX"*string(nsamples)
    
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    for θstart in -π:1:π, vstart in -1:1:1, ωstart in -1:1:1
        storage, _, _ = cartpole(Δt=Δtsim, θstart=θstart, vstart=vstart, ωstart=ωstart)
        dataset += storage
    end
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    xtrain, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    xtrain = reduce(hcat, xtrain)
    yv12 = [s[9] for s in xnext_train]
    yv22 = [s[22] for s in xnext_train]
    yv23 = [s[23] for s in xnext_train]
    yω21 = [s[24] for s in xnext_train]
    ytrain = [yv12, yv22, yv23, yω21]
    xtest_t0, _, xtest_tk = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    mechanism = cartpole(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism

    # Load checkpoints
    nprocessed, kstep_mse, params = get_hpsearch_params(_loadcheckpoint)


    # Generate random parameters
    stdx = std(xtrain, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (50 ./(stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 1.) .* params for _ in 1:ntrials-1]
    push!(paramtuples, params)  # Make sure initial params are also included
    
    # Create config dictionary
    config = get_config(EXPERIMENT_ID, mechanism, xtrain, ytrain, xtest_t0, [], xtest_tk, paramtuples,
                        nprocessed, kstep_mse, params)

    # Launch parallel search
    parallelsearch(experimentCPMax, config)
end

function hyperparametersearchP1Min(ntrials, nsamples, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P1_MIN"*string(nsamples)

    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=Δtsim, θstart=θ)
        dataset += storage
    end
    mechanism = simplependulum2D(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    xtrain, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    xtrain = [max2mincoordinates(cstate, mechanism) for cstate in xtrain]
    xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]
    xtrain = reduce(hcat, xtrain)
    ytrain = [[s[2] for s in xnext_train]]
    xtest_t0, xtest_t1, xtest_tk = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]
    xtest_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1]
    # intentionally not converting xtest_tk since final comparison is done in maximal coordinates

    # Load checkpoints
    nprocessed, kstep_mse, params = get_hpsearch_params(_loadcheckpoint)

    # Generate random parameters
    stdx = std(xtrain, dims=2)
    stdx[stdx .== 0] .= 100
    params = [1.1, (10 ./(stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 0.999) .* params for _ in 1:ntrials-1]
    push!(paramtuples, params)  # Make sure initial params are also included

    # Create config dictionary
    config = get_config(EXPERIMENT_ID, mechanism, xtrain, ytrain, xtest_t0, xtest_t1, xtest_tk, paramtuples,
                        nprocessed, kstep_mse, params)

    # Launch parallel search
    parallelsearch(experimentP1Min, config)
end    

function hyperparametersearchP2Min(ntrials, nsamples, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P2_MIN"*string(nsamples)

    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = doublependulum2D(Δt=Δtsim, θstart=[θ1, θ2])
        dataset += storage
    end
    mechanism = doublependulum2D(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    xtrain, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    xtrain = [max2mincoordinates(cstate, mechanism) for cstate in xtrain]
    xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]
    xtrain = reduce(hcat, xtrain)
    yω1 = [s[2] for s in xnext_train]
    yω2 = [s[4] for s in xnext_train]
    ytrain = [yω1, yω2]
    xtest_t0, xtest_t1, xtest_tk = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]
    xtest_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1]
    # intentionally not converting xtest_tk since final comparison is done in maximal coordinates

    # Load checkpoints
    nprocessed, kstep_mse, params = get_hpsearch_params(_loadcheckpoint)

    # Generate random parameters
    stdx = std(xtrain, dims=2)
    stdx[stdx .== 0] .= 100
    params = [1.1, (1 ./(0.02 .*stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 0.999) .* params for _ in 1:ntrials-1]
    push!(paramtuples, params)  # Make sure initial params are also included

    # Create config dictionary
    config = get_config(EXPERIMENT_ID, mechanism, xtrain, ytrain, xtest_t0, xtest_t1, xtest_tk, paramtuples,
                        nprocessed, kstep_mse, params)

    # Launch parallel search
    parallelsearch(experimentP2Min, config)
end

function hyperparametersearchCPMin(ntrials, nsamples, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "CP_MIN"*string(nsamples)

    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    for θstart in -π:1:π, vstart in -1:1:1, ωstart in -1:1:1
        storage, _, _ = cartpole(Δt=Δtsim, θstart=θstart, vstart=vstart, ωstart=ωstart)
        dataset += storage
    end
    mechanism = cartpole(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    xtrain, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    xtrain = [max2mincoordinates(cstate, mechanism) for cstate in xtrain]
    xnext_train = [max2mincoordinates(cstate, mechanism) for cstate in xnext_train]
    xtrain = reduce(hcat, xtrain)
    yv = [s[2] for s in xnext_train]
    yω = [s[4] for s in xnext_train]
    ytrain = [yv, yω]
    xtest_t0, xtest_t1, xtest_tk = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    xtest_t0 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t0]
    xtest_t1 = [max2mincoordinates(cstate, mechanism) for cstate in xtest_t1]
    # intentionally not converting xtest_tk since final comparison is done in maximal coordinates

    # Load checkpoints
    nprocessed, kstep_mse, params = get_hpsearch_params(_loadcheckpoint)

    # Generate random parameters
    stdx = std(xtrain, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (50 ./(stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 0.999) .* params for _ in 1:ntrials-1]
    push!(paramtuples, params)  # Make sure initial params are also included

    # Create config dictionary
    config = get_config(EXPERIMENT_ID, mechanism, xtrain, ytrain, xtest_t0, xtest_t1, xtest_tk, paramtuples,
                        nprocessed, kstep_mse, params)

    # Launch parallel search
    parallelsearch(experimentCPMin, config)
end    

settings = Dict("nruns" => 1,
                "Δtsim" => 0.001,
                "ntestsets" => 5,
                "testsamples" => 1000,
                "simsteps" => 10)

for nsamples in [2] # , 4, 8, 16, 32, 64, 128, 256, 512]
    hyperparametersearchP1Max(settings, nsamples)
    # hyperparametersearchP2Max(settings, nsamples)
    # hyperparametersearchCPMax(settings, nsamples)
    #hyperparametersearchP1Min(settings, nsamples)
    #hyperparametersearchP2Min(settings, nsamples)
    #hyperparametersearchCPMin(settings, nsamples)
end
