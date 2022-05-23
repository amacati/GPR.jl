using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

"""
    Generic experiment body that can be used with and without dynamics.
"""
function _experiment_cp_max(config, id; meandynamics = false)
    traindfs, testdfs = config["datasets"]
    # Each thread operates on its own dataset -> no races
    # Use the dataset nr `id` and choose `nsamples` random training samples.
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["nsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]
    mechanism = cartpole(1, Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    
    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, config["Σ"], "CP", config["Δtsim"], mechanism.bodies[2].shape.rh[2])
    end
    # Create train and testsets
    xtrain_old = reduce(hcat, [CState(x) for x in traindf.sold])
    xtrain_curr = [CState(x) for x in traindf.scurr]
    vωindices = [9, 22, 23, 24]  # We only need to train vx1, vx2, vy2, and ωx2
    ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]    
    xtest_old = [CState(x) for x in testdf.sold]   

    # The hyperparameters for each GP are further optimized given the noisy training data
    predictedstates = Vector{CState{Float64, 2}}()
    params = config["params"]
    gps = Vector{GPE}()
    cache = MDCache()
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = meandynamics ? MeanDynamics(mechanism, getμ(vωindices), id, cache) : MeanZero()
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    # Predict the dynamics of the test set and return the errors
    projectionerror = 0
    getvω(μ) = return [SVector(0,μ[1],0), SVector(0,μ[2:3]...)], [SVector(zeros(3)...), SVector(μ[4],0,0)]
    for i in 1:length(xtest_old)
        predictedstate, perror = predictdynamics(mechanism, gps, xtest_old[i], config["simsteps"], getvω)
        push!(predictedstates, predictedstate)
        projectionerror += perror
    end
    return predictedstates, xtest_future_true, projectionerror/length(xtest_old)
end

"""
    A wrapper for experiments without mean dynamics.
"""
function experiment_cp_mz_max(config, id)
    return _experiment_cp_max(config, id, meandynamics=false)
end

"""
    A wrapper for experiments with mean dynamics.
"""
function experiment_cp_md_max(config, id)
    return _experiment_cp_max(config, id, meandynamics=true)
end
