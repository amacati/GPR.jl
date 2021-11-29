using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentNoisyFBMinSin(config, id)
    traindfs, testdfs = loaddatasets("FB")
    ΔJ = traindfs.ΔJ[id]
    m = traindfs.m[id]
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["nsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]    
    mechanism = fourbar(1; Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.xyz[3]

    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, config["Σ"], "FB", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [max2mincoordinates_fb(CState(x)) for x in traindf.sold]
    xtrain_old = reduce(hcat, [[sin(s[1]), s[2], sin(s[3]), s[4]] for s in xtrain_old])
    xtrain_curr = [max2mincoordinates_fb(CState(x)) for x in traindf.scurr]
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # ω11, ω31
    xtest_old = [max2mincoordinates_fb(CState(x)) for x in testdf.sold]

    predictedstates = Vector{CState{Float64,4}}()
    params = config["params"]
    gps = Vector{GPE}()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "FB", gps, xtest_old[i], config["simsteps"], usesin = true)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end