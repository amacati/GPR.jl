include("utils.jl")


function parallelsimP1Max()
        # Generate UUID
        EXPERIMENT_ID = "P1_MAX"*string(nsamples)
    
        # Generate Dataset
        Δtsim = 0.001
        ntestsets = 5
        dataset = Dataset()
        # TODO: use noise
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
        nprocessed, onestep_msevec, onestep_params = get_checkpoint_params(_loadcheckpoint)
    
        # Generate random parameters
        stdx = std(x_train, dims=2)
        stdx[stdx .== 0] .= 1000
        params = [100., (10 ./(stdx))...]
        paramtuples = [params .+ (5rand(length(params)) .- 0.999) .* params for _ in 1:ntrials-1]
        push!(paramtuples, params)  # Make sure initial params are also included
    
        # Create config dictionary
        config = get_config(EXPERIMENT_ID, mechanism, x_train, y_train, x_test, [], xresult_test, paramtuples,
                            nprocessed, onestep_msevec, onestep_params)
    
        # Launch parallel search
        parallelsearch(experimentP1Max, config)
    end
    