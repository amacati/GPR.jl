using ConstrainedDynamics: Mechanism

mutable struct ParallelConfig
    EXPERIMENT_ID::String
    mechanism::Mechanism
    X::Vector
    Y::Vector{Vector}
    initialstates::Vector
    experimentlength::Integer
    paramtuples::AbstractArray
    nprocessed::Integer
    onestep_msevec::Vector
    onestep_params::Vector{Vector}
    paramlock::Base.AbstractLock
    resultlock::Base.AbstractLock

    function ParallelConfig(experimentid, mechanism, X, Y, initialstates, experimentlength, paramtuples, _loadcheckpoint)
        nprocessed = 0
        onestep_msevec = []
        onestep_params = []
        if _loadcheckpoint
            success, checkpointdict = loadcheckpoint(EXPERIMENT_ID)
            if success
                nprocessed = checkpointdict["nprocessed"]
                onestep_msevec = checkpointdict["onestep_msevec"]
                onestep_params = checkpointdict["onestep_params"]
            else
                @warn("No previous checkpoints found, search starts at 0.")
            end
        end
        new(experimentid, mechanism, X, Y, initialstates, experimentlength, paramtuples, nprocessed, onestep_msevec, onestep_params,
            ReentrantLock(), ReentrantLock())
    end
end

function parallelsearch(experiment, config)
    tstart = time()    
    Threads.@threads for _ in config.nprocessed+1:length(config.paramtuples)
        # Get hyperparameters (threadsafe)
        success, params = _getparams(config)  # Threadsafe
        !success && continue
        _checkpoint(config, tstart)
        # Main experiment
        storage = nothing  # Define in outer scope
        try
            storage = experiment(config, params)
        catch e
            # display(e)
        end
        lock(config.resultlock)
        # Writing the results
        try
            storage !== nothing ? onestep_mse = onesteperror(mechanism, storage) : onestep_mse = Inf
            push!(config.onestep_msevec, onestep_mse)
            push!(config.onestep_params, [params...])
        catch e
            display(e)
        finally
            unlock(config.resultlock)
        end
    end
    checkpointdict = Dict("nprocessed" => config.nprocessed, "onestep_msevec" => config.onestep_msevec,
                          "onestep_params" => config.onestep_params)
    checkpoint(config.EXPERIMENT_ID*"_FINAL", checkpointdict)
    println("Best one step mean squared error: $(minimum(config.onestep_msevec))")
end

function _getparams(config::ParallelConfig)
    lock(config.paramlock)
    try
        @assert config.nprocessed < length(config.paramtuples)
        params = config.paramtuples[config.nprocessed+1]
        config.nprocessed += 1
        return true, params
    catch e
        display(e)
        return false, []
    finally
        unlock(config.paramlock)
    end
end

function _checkpoint(config, tstart)
    if config.nprocessed % 10 == 0
        println("Processing job $(config.nprocessed)/$(length(config.paramtuples))")
        Δt = time() - tstart
        secs = Int(round(Δt*(length(config.paramtuples)/config.nprocessed-1)))
        hours = div(secs, 3600)
        minutes = div(secs-hours*3600, 60)
        secs -= (hours*3600 + minutes * 60)
        println("Estimated time to completion: $(hours)h, $(minutes)m, $(secs)s")
        if config.nprocessed % 100 == 0
            checkpointdict = Dict("nprocessed" => config.nprocessed, "onestep_msevec" => config.onestep_msevec,
                                "onestep_params" => config.onestep_params)
            checkpoint(config.EXPERIMENT_ID, checkpointdict)
        end
    end
end