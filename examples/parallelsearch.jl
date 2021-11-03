using ConstrainedDynamics: Mechanism, Storage
using Suppressor


function parallelsearch(experiment, config)
    tstart = time()
    Threads.@threads for _ in config["nprocessed"]+1:length(config["paramtuples"])
        # Get hyperparameters (threadsafe)
        success, params, jobid = _getparams(config)  # Threadsafe
        !success && continue
        _checkpoint(config, tstart, jobid)  # Threadsafe
        # Main experiment
        predictedstates = nothing  # Define in outer scope
        try
            predictedstates = experiment(config, params)  # GaussianProcesses.optimize! spams exceptions
        catch e
            display(e)
            throw(e)
        end
        lock(config["resultlock"])
        # Writing the results
        try
            predictedstates !== nothing ? onestep_mse = simulationerror(config["xresult_test"], predictedstates) : onestep_mse = Inf
            push!(config["onestep_msevec"], onestep_mse)
            push!(config["onestep_params"], [params...])
        catch e
            display(e)
            throw(e)
        finally
            unlock(config["resultlock"])
        end
    end  # End of threaded program
    checkpointdict = Dict("nprocessed" => config["nprocessed"], "onestep_msevec" => config["onestep_msevec"],
                          "onestep_params" => config["onestep_params"])
    checkpoint(config["EXPERIMENT_ID"]*"_FINAL", checkpointdict)
    println("Best one step mean squared error: $(minimum(config["onestep_msevec"]))")
end

function _getparams(config)
    lock(config["paramlock"])
    try
        @assert config["nprocessed"] < length(config["paramtuples"])
        params = config["paramtuples"][config["nprocessed"]+1]
        config["nprocessed"] += 1
        return true, params, config["nprocessed"]
    catch e
        # display(e)
        return false, [], 0
    finally
        unlock(config["paramlock"])
    end
end

function _checkpoint(config, tstart, jobid)
    if jobid % 10 == 0
        println("Processing job $(jobid)/$(length(config["paramtuples"]))")
        Δt = time() - tstart
        secs = Int(round(Δt*(length(config["paramtuples"])/jobid-1)))
        hours = div(secs, 3600)
        minutes = div(secs-hours*3600, 60)
        secs -= (hours*3600 + minutes * 60)
        println("Estimated time to completion: $(hours)h, $(minutes)m, $(secs)s")
        if jobid % 100 == 0
            lock(config["checkpointlock"])
            try
                checkpointdict = Dict("nprocessed" => jobid, "onestep_msevec" => config["onestep_msevec"],
                                    "onestep_params" => config["onestep_params"])
                checkpoint(config["EXPERIMENT_ID"], checkpointdict)
            catch
            finally
                unlock(config["checkpointlock"])
            end
        end
    end
end