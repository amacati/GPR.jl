using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentP2Min(config)
    mechanism = deepcopy(config["mechanism"])
    # Sample from dataset
    xtrain_old = reduce(hcat, [max2mincoordinates(CState(x),mechanism) for x in config["traindf"].sold])
    xtrain_curr = [max2mincoordinates(CState(x), mechanism) for x in config["traindf"].scurr]
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # ω11, ω21
    xtest_old = [max2mincoordinates(CState(x),mechanism) for x in config["testdf"].sold]
    xtest_future = [CState(x) for x in config["testdf"].sfuture]
    # intentionally not converting xtest_future since final comparison is done in maximal coordinates

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1.1, (50 ./stdx)...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{CState{Float64,2}}()
    gps = Vector{GPE}()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "P2", gps, xtest_old[i], config["simsteps"])
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future, params
end
