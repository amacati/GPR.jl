using ConstrainedDynamics: Mechanism, Storage
using Suppressor


function parallelsim(experiment, config)
    tstart = time()
    for jobid in config["nprocessed"]+1:config["nruns"]  # Threads.@threads 
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
        _checkpoint(config, tstart, jobid)  # Threadsafe
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
            predictedstates !== nothing ? onestep_mse = simulationerror(xresult_test, predictedstates) : onestep_mse = nothing
            push!(config["onestep_msevec"], onestep_mse)
        catch e
            display(e)
            throw(e)
        finally
            unlock(config["resultlock"])
        end
    end  # End of threaded program
    checkpointdict = Dict("nprocessed" => config["nprocessed"], "onestep_msevec" => config["onestep_msevec"])
    checkpoint(config["EXPERIMENT_ID"]*"_FINAL", checkpointdict)
    println("Best one step mean squared error: $(minimum(config["onestep_msevec"]))")
end

function _checkpoint(config, tstart, jobid)
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
                checkpointdict = Dict("nprocessed" => jobid, "onestep_msevec" => config["onestep_msevec"])
                checkpoint(config["EXPERIMENT_ID"], checkpointdict)
            catch
            finally
                unlock(config["checkpointlock"])
            end
        end
    end
end