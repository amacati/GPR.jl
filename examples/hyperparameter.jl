include("parallelsearch.jl")
include("generatedata.jl")
include("maximal_coordinates/2Dpendulum.jl")
include("maximal_coordinates/2DdoublePendulum.jl")
include("maximal_coordinates/cartpole.jl")
include("utils.jl")
include("dataset.jl")

function hyperparametersearchP1Max(ntrials, nsamples, loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P1_MAX"*string(nsamples)
    
    # Generate Dataset
    Δtsim = 0.001
    ntestsets = 5
    dataset = Dataset()
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=Δtsim, θstart=θ)
        dataset += storage
    end
    testsets = Integer.(round.(collect(range(1, stop=length(dataset.storages), length=ntestsets))))
    x_train, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    x_train = reduce(hcat, x_train)
    yv2 = [s[9] for s in xnext_train]
    yv3 = [s[10] for s in xnext_train]
    yω = [s[11] for s in xnext_train]
    y_train = [yv2, yv3, yω]
    x_test, _, xresult_test = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    mechanism = simplependulum2D(Δt=0.01, θstart=0.)[2]  # Reset Δt to 0.01 in mechanism
    
    # Load checkpoints
    nprocessed = 0
    onestep_msevec = []
    onestep_params = []
    if loadcheckpoint
        success, checkpointdict = loadcheckpoint(EXPERIMENT_ID)
        if success
            nprocessed = checkpointdict["nprocessed"]
            onestep_msevec = checkpointdict["onestep_msevec"]
            onestep_params = checkpointdict["onestep_params"]
        else
            @warn("No previous checkpoints found, search starts at 0.")
        end
    end

    # Generate random parameters
    stdx = std(x_train, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (10 ./(stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 0.999) .* params for _ in 1:ntrials-1]
    push!(paramtuples, params)  # Make sure initial params are also included

    # Create config dictionary
    config = Dict("EXPERIMENT_ID"=>EXPERIMENT_ID,
                  "mechanism"=>mechanism, 
                  "x_train"=>x_train, 
                  "y_train"=>y_train,
                  "x_test"=>x_test,
                  "xnext_test"=>[],
                  "xresult_test"=>xresult_test, 
                  "paramtuples"=>paramtuples, 
                  "nprocessed"=>nprocessed, 
                  "onestep_msevec"=>onestep_msevec, 
                  "onestep_params"=>onestep_params, 
                  "paramlock"=>ReentrantLock(), 
                  "checkpointlock"=>ReentrantLock(), 
                  "resultlock"=>ReentrantLock())
    # Launch parallel search
    parallelsearch(experimentP1Max, config)
end

function hyperparametersearchP2Max(ntrials, nsamples, loadcheckpoint=false)
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
    x_train, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    x_train = reduce(hcat, x_train)
    yv12 = [s[9] for s in xnext_train]
    yv13 = [s[10] for s in xnext_train]
    yv22 = [s[22] for s in xnext_train]
    yv23 = [s[23] for s in xnext_train]
    yω11 = [s[11] for s in xnext_train]
    yω21 = [s[24] for s in xnext_train]
    y_train = [yv12, yv13, yv22, yv23, yω11, yω21]
    x_test, _, xresult_test = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    mechanism = doublependulum2D(Δt=0.01, θstart=[0, 0])[2]  # Reset Δt to 0.01 in mechanism
    
    # Load checkpoints
    nprocessed = 0
    onestep_msevec = []
    onestep_params = []
    if loadcheckpoint
        success, checkpointdict = loadcheckpoint(EXPERIMENT_ID)
        if success
            nprocessed = checkpointdict["nprocessed"]
            onestep_msevec = checkpointdict["onestep_msevec"]
            onestep_params = checkpointdict["onestep_params"]
        else
            @warn("No previous checkpoints found, search starts at 0.")
        end
    end

    # Generate random parameters
    stdx = std(x_train, dims=2)
    stdx[stdx .== 0] .= 100
    params = [1.1, (1 ./(0.02 .*stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 0.999) .* params for _ in 1:ntrials-1]
    push!(paramtuples, params)  # Make sure initial params are also included

    # Create config dictionary
    config = Dict("EXPERIMENT_ID"=>EXPERIMENT_ID,
                  "mechanism"=>mechanism, 
                  "x_train"=>x_train, 
                  "y_train"=>y_train,
                  "x_test"=>x_test,
                  "xnext_test"=>[],
                  "xresult_test"=>xresult_test, 
                  "paramtuples"=>paramtuples, 
                  "nprocessed"=>nprocessed, 
                  "onestep_msevec"=>onestep_msevec, 
                  "onestep_params"=>onestep_params, 
                  "paramlock"=>ReentrantLock(), 
                  "checkpointlock"=>ReentrantLock(), 
                  "resultlock"=>ReentrantLock())
    # Launch parallel search
    parallelsearch(experimentP2Max, config)
end

function hyperparametersearchCPMax(ntrials, nsamples, loadcheckpoint=false)
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
    x_train, xnext_train, _ = sampledataset(dataset, nsamples, Δt = Δtsim, exclude = testsets)
    x_train = reduce(hcat, x_train)
    yv12 = [s[9] for s in xnext_train]
    yv22 = [s[22] for s in xnext_train]
    yv23 = [s[23] for s in xnext_train]
    yω21 = [s[24] for s in xnext_train]
    y_train = [yv12, yv22, yv23, yω21]
    x_test, _, xresult_test = sampledataset(dataset, 1000, Δt = Δtsim, exclude = [i for i in 1:length(dataset.storages) if !(i in testsets)])
    mechanism = cartpole(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism

    # Load checkpoints
    nprocessed = 0
    onestep_msevec = []
    onestep_params = []
    if loadcheckpoint
        success, checkpointdict = loadcheckpoint(EXPERIMENT_ID)
        if success
            nprocessed = checkpointdict["nprocessed"]
            onestep_msevec = checkpointdict["onestep_msevec"]
            onestep_params = checkpointdict["onestep_params"]
        else
            @warn("No previous checkpoints found, search starts at 0.")
        end
    end

    # Generate random parameters
    stdx = std(x_train, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (50 ./(stdx))...]
    paramtuples = [params .+ (5rand(length(params)) .- 1.) .* params for _ in 1:ntrials]
    push!(paramtuples, params)  # Make sure initial params are also included
    
    # Create config dictionary
    config = Dict("EXPERIMENT_ID"=>EXPERIMENT_ID,
                  "mechanism"=>mechanism, 
                  "x_train"=>x_train, 
                  "y_train"=>y_train,
                  "x_test"=>x_test,
                  "xnext_test"=>[],
                  "xresult_test"=>xresult_test, 
                  "paramtuples"=>paramtuples, 
                  "nprocessed"=>nprocessed, 
                  "onestep_msevec"=>onestep_msevec, 
                  "onestep_params"=>onestep_params, 
                  "paramlock"=>ReentrantLock(), 
                  "checkpointlock"=>ReentrantLock(), 
                  "resultlock"=>ReentrantLock())
    # Launch parallel search
    parallelsearch(experimentCPMax, config)
end


ntrials = 100
for nsamples in [8, 16, 32, 64, 128, 256, 512]
    hyperparametersearchP1Max(ntrials, nsamples)
    hyperparametersearchP2Max(ntrials, nsamples)
    hyperparametersearchCPMax(ntrials, nsamples)
end
