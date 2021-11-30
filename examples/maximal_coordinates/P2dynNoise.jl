using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyP2Max(config, id)
    traindfs, testdfs = config["datasets"]  # Each thread operates on its own dataset -> no races
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["nsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]    
    mechanism = doublependulum2D(1; Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, config["Σ"], "P2", config["Δtsim"], mechanism.bodies[1].shape.xyz[3], mechanism.bodies[2].shape.xyz[3])
    end
    # Create train and testsets
    xtrain_old = reduce(hcat, [CState(x) for x in traindf.sold])
    xtrain_curr = [CState(x) for x in traindf.scurr]
    vωindices = [9, 10, 22, 23, 11, 24]  # v12, v13, v22, v23, ω11, ω21
    ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]
    xtest_old = [CState(x) for x in testdf.sold]

    predictedstates = Vector{CState{Float64,2}}()
    params = config["params"]
    gps = Vector{GPE}()
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, getμ(vωindices[id]))
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    t2 = time()
    projectionerror = 0
    getvω(μ) = return [SVector(0,μ[1:2]...), SVector(0,μ[3:4]...)], [SVector(μ[5],0,0), SVector(μ[6],0,0)]
    for i in 1:length(xtest_old)
        predictedstate, perror = predictdynamics(mechanism, gps, xtest_old[i], config["simsteps"], getvω)
        push!(predictedstates, predictedstate)
        projectionerror += perror
    end
    return predictedstates, xtest_future_true, projectionerror/length(xtest_old)
end
