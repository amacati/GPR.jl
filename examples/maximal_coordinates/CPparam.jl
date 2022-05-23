using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

"""
    Learn the dynamics from perfect data and calculate the error compared to the test set. The hyperparameters
    for each run are chosen at random from a reasonable start distribution that is realted to the training data.
    This `reasonable` distribution has been found by trial and error and visual inspection of the dynamics predicted
    by the GPs.
"""
function experimentCPMax(config, id)
    traindfs, testdfs = config["datasets"]  # Each thread operates on its own dataset -> no races
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["trainsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]
    mechanism = cartpole(1, Δt=0.01, threadlock=config["mechanismlock"])[2]

    xtrain_old = reduce(hcat, [CState(x) for x in traindf.sold])
    xtrain_curr = [CState(x) for x in traindf.scurr]
    vωindices = [9, 22, 23, 24]
    ytrain = [[cs[i] for cs in xtrain_curr] for i in vωindices]
    xtest_old = [CState(x) for x in testdf.sold]
    xtest_future = [CState(x) for x in testdf.sfuture]

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [100., (50 ./(stdx))...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{CState{Float64, 2}}()
    gps = Vector{GPE}()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    getvω(μ) = return [SVector(0,μ[1],0), SVector(0,μ[2:3]...)], [SVector(zeros(3)...), SVector(μ[4],0,0)]
    for i in 1:length(xtest_old)
        predictedstate, _ = predictdynamics(mechanism, gps, xtest_old[i], config["simsteps"], getvω)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future, params
end
