function parallelsearch(experiment, config)
    tstart = time()
    Threads.@threads for jobid in config["nprocessed"]+1:length(config["paramtuples"])
        # Get hyperparameters (threadsafe)
        lock(config["paramlock"])
        try
            @assert config["nprocessed"] < config["nruns"]
            config["nprocessed"] += 1
        catch e
            continue
        finally
            unlock(config["paramlock"])
        end
        checkpoint_hpsearch(config, tstart, jobid)  # Threadsafe
        # Main experiment
        params, predictedstates, test_tk = nothing, nothing, nothing  # Define in outer scope
        try
            params, predictedstates, xtest_tk = experiment(config)  # GaussianProcesses.optimize! spams exceptions
        catch e
            display(e)
            throw(e)
        end
        lock(config["resultlock"])
        # Writing the results
        try
            predictedstates !== nothing ? kstep_mse = simulationerror(test_tk, predictedstates) : kstep_mse = Inf
            push!(config["kstep_mse"], kstep_mse)
            push!(config["params"], params)
        catch e
            display(e)
            throw(e)
        finally
            unlock(config["resultlock"])
        end
    end  # End of threaded program
    # Load results if available
    checkpoint = Dict()
    try
        checkpoint = loadcheckpoint("params")  # Fails if file not found -> Create new checkpoint dict
    catch
    end
    checkpoint[config["EXPERIMENT_ID"]] = Dict("nprocessed" => jobid, "params"=> config["params"], "kstep_mse" => config["kstep_mse"])  # Append to results if any
    savecheckpoint("params", checkpoint)
    println("Best k-step mean squared error: $(minimum(config["kstep_mse"]))")
end

function checkpoint_hpsearch(config, tstart, jobid)
    if jobid % 10 == 0
        println("Processing job $(jobid)/$(length(config["paramtuples"]))")
        Δt = time() - tstart
        secs = Int(round(Δt*(length(config["paramtuples"])/jobid-1)))
        hours = div(secs, 3600)
        minutes = div(secs-hours*3600, 60)
        secs -= (hours*3600 + minutes * 60)
        println("Estimated time to completion: $(hours)h, $(minutes)m, $(secs)s")
        if jobid % 20 == 0
            lock(config["checkpointlock"])
            try
                # Load results if available
                checkpoint = Dict()
                try
                    checkpoint = loadcheckpoint("params")  # Fails if file not found -> Create new checkpoint dict
                catch
                end
                checkpoint[config["EXPERIMENT_ID"]] = Dict("nprocessed" => jobid, "params"=> config["params"], "kstep_mse" => config["kstep_mse"])  # Append to results if any
                savecheckpoint("params", checkpoint)
            catch
            finally
                unlock(config["checkpointlock"])
            end
        end
    end
end