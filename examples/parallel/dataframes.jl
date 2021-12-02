function generate_dataframes(config, exp1, exp2, exptest)
    max_trajectories = 100
    Ntrajectories = min(config["trainsamples"], max_trajectories)
    Ntrajectories_test = min(config["testsamples"], max_trajectories)
    Ntrajectorysamples = Int(ceil(config["trainsamples"]/max_trajectories))
    scaling = Int(0.01/config["Δtsim"])
    samplerange = 1:(2*Int(1/config["Δtsim"]) - scaling)
    traindf = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}())
    threadlock = ReentrantLock()  # Push to df not atomic
    Threads.@threads for _ in 1:div(Ntrajectories, 2)
        storage = _run_experiment!(exp1)
        _pushsamples!(storage, traindf, Ntrajectorysamples, samplerange, [0, scaling], scaling, threadlock)
    end
    Threads.@threads for _ in 1:div(Ntrajectories, 2)
        storage = _run_experiment!(exp2)
        _pushsamples!(storage, traindf, Ntrajectorysamples, samplerange, [0, scaling], scaling, threadlock)
    end
    testdf = DataFrame(sold = Vector{Vector{State}}(), sfuture = Vector{Vector{State}}())
    samplerange = 1:(2*Int(1/config["Δtsim"]) - scaling*(config["simsteps"]+1))
    Threads.@threads for _ in 1:Ntrajectories_test
        storage = _run_experiment!(exptest)
        _pushsamples!(storage, testdf, Int(ceil(config["testsamples"]/Ntrajectories_test)), samplerange, [0, scaling*(config["simsteps"]+1)], scaling, threadlock)
    end
    return traindf, testdf
end

function _run_experiment!(experiment; maxruns = 10)
    for _ in 1:maxruns  # Retry experiment if simulation fails
        try
            return experiment()
        catch e
            @warn "Experiment failed, retrying..."
            display(e)
            continue
        end
    end
    throw(ErrorException("Experiment failed to execute $maxruns times"))
end

function _pushsamples!(storage, df, nsamples, samplerange, indexoffset, scaling, threadlock)
    indiceset = Set()
    for _ in 1:nsamples
        j = 0
        while true
            j = rand(samplerange)  # End of storage - required steps
            !any([j in ind-2scaling:ind+2scaling for ind in indiceset]) && break  # Sample j outside of existing indices
        end
        lock(threadlock)
        try
            push!(df, [getStates(storage, j+offset) for offset in indexoffset])
            push!(indiceset, j)
        finally
            unlock(threadlock)
        end
    end
end
