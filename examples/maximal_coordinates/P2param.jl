using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics
using DataFrames


function experimentP2Max(config, id)
    traindfs, testdfs = config["datasets"]  # Each thread operates on its own dataset -> no races
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["trainsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]
    mechanism = doublependulum2D(1, Δt=0.01, threadlock=config["mechanismlock"])[2]

    xtrain_old = reduce(hcat, [tocstate(x) for x in traindf.sold])
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    vωindices = [9, 10, 22, 23, 11, 24]
    ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]
    xtest_old = [tocstate(x) for x in testdf.sold]
    xtest_future = [tocstate(x) for x in testdf.sfuture]

    # Sample random parameters
    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1.1, (50 ./stdx)...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector{GPE}()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    getvω(μ) = return [SVector(0,μ[1:2]...), SVector(0,μ[3:4]...)], [SVector(μ[5],0,0), SVector(μ[6],0,0)]
    for i in 1:length(xtest_old)
        predictedstate, _ = predictdynamics(mechanism, gps, xtest_old[i], config["simsteps"], getvω)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future, params
end
