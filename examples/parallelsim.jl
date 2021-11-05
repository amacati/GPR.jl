function parallelsim(experiment, config)
    tstart = time()
    Threads.@threads for jobid in config["nprocessed"]+1:config["nruns"]  # Threads.@threads 
        # Increment nprocessed (threadsafe)
        lock(config["paramlock"])
        try
            @assert config["nprocessed"] < config["nruns"]
            config["nprocessed"] += 1
        catch e
            continue
        finally
            unlock(config["paramlock"])
        end
        params = config["params"]
        checkpoint_sim(config, tstart, jobid)  # Threadsafe
        # Main experiment
        predictedstates = nothing  # Define in outer scope
        xresult_test = nothing
        try
            predictedstates, xresult_test = experiment(config, params)  # GaussianProcesses.optimize! spams exceptions
        catch e
            display(e)
            throw(e)
        end
        lock(config["resultlock"])
        # Writing the results
        try
            predictedstates !== nothing ? kstep_mse = simulationerror(xresult_test, predictedstates) : kstep_mse = nothing
            push!(config["kstep_mse"], kstep_mse)
        catch e
            display(e)
            throw(e)
        finally
            unlock(config["resultlock"])
        end
    end  # End of threaded program
    results = Dict()
    try
        results = loadcheckpoint("noisy_final")  # Fails if file not found -> Create new results dict
    catch
    end
    results[config["EXPERIMENT_ID"]] = Dict("nprocessed" => config["nprocessed"], "kstep_mse" => config["kstep_mse"])
    savecheckpoint("noisy_final", results)
    println("Best k-step mean squared error: $(minimum(config["kstep_mse"]))")
end

function checkpoint_sim(config, tstart, jobid)
    if jobid % 10 == 0
        println("Processing job $(jobid)/$(config["nruns"])")
        Δt = time() - tstart
        secs = Int(round(Δt*(config["nruns"]/jobid-1)))
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
                    checkpoint = loadcheckpoint("noisy")  # Fails if file not found -> Create new checkpoint dict
                catch
                end
                checkpoint[config["EXPERIMENT_ID"]] = Dict("nprocessed" => jobid, "kstep_mse" => config["kstep_mse"])  # Append to results if any
                savecheckpoint("noisy", checkpoint)
            catch
            finally
                unlock(config["checkpointlock"])
            end
        end
    end
end