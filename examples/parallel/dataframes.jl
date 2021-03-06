using DataFrames

"""
    Generate up to 100 trajectories for the training set and 100 for the test set. Randomly sample these trajectories until the desired dataset size has been reached.
    If generate_uniform, also generate a dataset where the samples are chosen in uniform intervals from each trajectory. 
"""
function generate_dataframes(config, exp1, exp2, exptest; generate_uniform = false)
    # Calculate the number of trajectories and samples per trajectory
    max_trajectories = 100
    Ntrajectories = min(config["trainsamples"], max_trajectories)
    Ntrajectories_test = min(config["testsamples"], max_trajectories)
    Ntrajectorysamples = Int(ceil(config["trainsamples"]/max_trajectories))
    scaling = Int(0.01/config["Δtsim"])
    samplerange = 1:(2*Int(1/config["Δtsim"]) - scaling)
    traindf = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}())
    generate_uniform && (traindf_uniform = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}()))
    threadlock = ReentrantLock()  # Push to df not atomic -> push requires lock
    # Generate the trajectories in parallel and push random samples into the dataset
    Threads.@threads for _ in 1:div(Ntrajectories, 2)
        storage = _run_experiment!(exp1)
        _pushsamples!(storage, traindf, Ntrajectorysamples, samplerange, [0, scaling], scaling, threadlock)
        generate_uniform && _pushuniformsamples!(storage, traindf_uniform, Ntrajectorysamples, samplerange, [0, scaling], threadlock)
    end
    Threads.@threads for _ in 1:div(Ntrajectories, 2)
        storage = _run_experiment!(exp2)
        _pushsamples!(storage, traindf, Ntrajectorysamples, samplerange, [0, scaling], scaling, threadlock)
        generate_uniform && _pushuniformsamples!(storage, traindf_uniform, Ntrajectorysamples, samplerange, [0, scaling], threadlock)
    end
    testdf = DataFrame(sold = Vector{Vector{State}}(), sfuture = Vector{Vector{State}}())
    generate_uniform && (testdf_uniform = DataFrame(sold = Vector{Vector{State}}(), sfuture = Vector{Vector{State}}()))
    samplerange = 1:(2*Int(1/config["Δtsim"]) - scaling*(config["simsteps"]+1))
    Threads.@threads for _ in 1:Ntrajectories_test
        storage = _run_experiment!(exptest)
        _pushsamples!(storage, testdf, Int(ceil(config["testsamples"]/Ntrajectories_test)), samplerange, [0, scaling*(config["simsteps"]+1)], scaling, threadlock)
        generate_uniform && _pushuniformsamples!(storage, testdf_uniform, Int(ceil(config["testsamples"]/Ntrajectories_test)), samplerange, [0, scaling*(config["simsteps"]+1)], threadlock)
    end
    generate_uniform && return traindf, testdf, traindf_uniform, testdf_uniform
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

"""
    Push random samples from a trajectory to the dataset. Threadsafe. Samples are guaranteed to not 
    intersect each other (next_state_1 is always earlier than start_state_2).
"""
function _pushsamples!(storage, df, nsamples, samplerange, indexoffset, scaling, threadlock)
    @assert (4scaling+1)*nsamples < length(samplerange)
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

"""
    Push samples in uniform intervals to the dataset. Threadsafe.
"""
function _pushuniformsamples!(storage, df, nsamples, samplerange, indexoffset, threadlock)
    @assert length(samplerange) > nsamples
    indiceset = Set()
    indices = nsamples > 1 ? [samplerange[i*div(length(samplerange),nsamples)] for i in 1:nsamples] : div(length(samplerange),2)
    for j in indices
        lock(threadlock)
        try
            push!(df, [getStates(storage, j+offset) for offset in indexoffset])
            push!(indiceset, j)
        finally
            unlock(threadlock)
        end
    end
end
