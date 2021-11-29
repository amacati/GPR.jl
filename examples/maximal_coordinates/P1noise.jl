using GaussianProcesses
using DataFrames
using Random
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentNoisyP1Max(config, id)
    traindfs, testdfs = loaddatasets("P1")
    ΔJ = traindfs.ΔJ[id]
    m = traindfs.m[id]
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["nsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]
    mechanism = simplependulum2D(1, Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M

    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, config["Σ"], "P1", config["Δtsim"], mechanism.bodies[1].shape.rh[2])
    end
    # Create train and testsets
    xtrain_old = reduce(hcat, [CState(x) for x in traindf.sold])
    xtrain_curr = [CState(x) for x in traindf.scurr]
    vωindices = [9, 10, 11]
    ytrain = [[cs[i] for cs in xtrain_curr] for i in vωindices]
    xtest_old = [CState(x) for x in testdf.sold]

    predictedstates = Vector{CState{Float64,1}}()
    params = config["params"]
    gps = Vector{GPE}()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    projectionerror = 0
    getvω(μ) = return [SVector(0,μ[1:2]...)], [SVector(μ[3],0,0)]
    for i in 1:length(xtest_old)
        predictedstate, perror = predictdynamics(mechanism, gps, xtest_old[i], config["simsteps"], getvω)
        push!(predictedstates, predictedstate)
        projectionerror += perror
    end

    return predictedstates, xtest_future_true, projectionerror/length(xtest_old)
end
