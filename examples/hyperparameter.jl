include("parallelsearch.jl")
include("generatedata.jl")
include("maximal_coordinates/2Dpendulum.jl")
include("utils.jl")
include("dataset.jl")

function hyperparametersearchP1Max(ntrials, nsamples, loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P1_MAX"*string(nsamples)
    
    # Generate Dataset
    Δtsim = 0.001
    testsets = [3, 7, 9, 20]
    dataset = Dataset()
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=Δtsim, θstart=θ)
        dataset += storage
    end
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
    parallelsearch(experiment, config)
end

ntrials = 100
for nsamples in [10, 32, 64, 128, 256, 512]
    hyperparametersearchP1Max(ntrials, nsamples)
end