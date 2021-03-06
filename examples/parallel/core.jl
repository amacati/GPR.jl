"""
    Checkpoints results every 50 experiments. Locks needed to avoid data races.
"""
function checkpointgeneric(etype::String, config::Dict, jobid::Integer, checkpointcallback!::Function)
    if jobid % 50 == 0
        @info ("Main Thread: Processing job $(jobid)/$(config["nruns"]), jobID $(config["EXPERIMENT_ID"])")
        lock(config["checkpointlock"])
        try
            checkpoint = Dict()
            try
                checkpoint = loadcheckpoint(etype)  # Fails if file not found -> Create new checkpoint dict
            catch
            end
            checkpointcallback!(checkpoint, config)
            savecheckpoint(etype, checkpoint)
        catch e
            # throw(e)
        finally
            unlock(config["checkpointlock"])
        end
    end
end

"""
    Run experiments on multiple worker nodes and safely handle results.
"""
function parallelrun(etype::String, experiment::Function, config::Dict, checkpointcallback!::Function, resultcallback!::Function, finalcallback!::Function)
    Threads.@threads for jobid in config["nprocessed"]+1:config["nruns"] 
        # Increment nprocessed (threadsafe)
        lock(config["paramlock"])
        try
            @assert config["nprocessed"] < config["nruns"]
            config["nprocessed"] += 1
        catch
            continue
        finally
            unlock(config["paramlock"])
        end
        # Main experiment
        result = nothing  # Define in outer scope to avoid undefined variable in case of experiment failure
        try
            result = experiment(config, jobid)  # GaussianProcesses.optimize! spams exceptions
        catch e
            display(e)
            # throw(e)
        end
        lock(config["resultlock"])
        # Write the results
        try
            resultcallback!(result, config)
        catch e
            display(e)
            # throw(e)
        finally
            unlock(config["resultlock"])
        end
        checkpointgeneric(etype, config, jobid, checkpointcallback!)  # Threadsafe
    end  # End of threaded program
    results = Dict()
    try
        results = loadcheckpoint(etype*"_final")  # Fails if file not found -> Create new results dict
    catch
    end
    finalcallback!(results, config)
    savecheckpoint(etype*"_final", results)
    @info ("Main Thread: Parallel run $(config["EXPERIMENT_ID"]) finished successfully.")
end

"""
    Define callbacks for checkpointing and result handling for noise experiments and execute the experiment on multiple workers.
"""
function parallelsim(experiment::Function, config::Dict; idmod::String = "")
    etype = "noisy" * idmod  # Mean dynamics, sin need different ID
    function checkpointcallback!(checkpoint::Dict, config::Dict)
        checkpoint[config["EXPERIMENT_ID"]] = Dict("nprocessed" => config["nprocessed"], "kstep_mse" => config["kstep_mse"])  # Append to results if any
    end
        
    function resultcallback!(result, config::Dict)
        result[1] !== nothing ? kstep_mse = simulationerror(result[2], result[1]) : kstep_mse = nothing
        (length(result) == 3 && result[3] !== nothing) ? projectionerror = result[3] : projectionerror = 0
        push!(config["kstep_mse"], kstep_mse)
        push!(config["projectionerror"], projectionerror)
    end

    function finalcallback!(results, config::Dict)
        results[config["EXPERIMENT_ID"]] = Dict("nprocessed" => config["nprocessed"], "kstep_mse" => config["kstep_mse"], "projectionerror" => config["projectionerror"])
    end
    parallelrun(etype, experiment, config, checkpointcallback!, resultcallback!, finalcallback!)
end

"""
    Define callbacks for checkpointing and result handling for the parameter search and execute the search on multiple workers.
"""
function parallelsearch(experiment::Function, config::Dict)
    etype = "params"

    function checkpointcallback!(checkpoint::Dict, config::Dict)
        checkpoint[config["EXPERIMENT_ID"]] = Dict("nprocessed" => config["nprocessed"], "params"=> config["params"], "kstep_mse" => config["kstep_mse"])  # Append to results if any
    end
        
    function resultcallback!(result, config::Dict)
        result[1] !== nothing ? kstep_mse = simulationerror(result[2], result[1]) : kstep_mse = Inf
        push!(config["kstep_mse"], kstep_mse)
        push!(config["params"], result[3])
    end

    function finalcallback!(results, config::Dict)    
        results[config["EXPERIMENT_ID"]] = Dict("nprocessed" => config["nprocessed"], "params" => config["params"], "kstep_mse" => config["kstep_mse"])
    end

    parallelrun(etype, experiment, config, checkpointcallback!, resultcallback!, finalcallback!)
end