using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentFBMax(config, id)
    traindfs, testdfs = config["datasets"]  # Each thread operates on its own dataset -> no races
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["trainsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]
    mechanism = fourbar(1, Δt=0.01, threadlock=config["mechanismlock"])[2]

    xtrain_old = reduce(hcat, [CState(x) for x in traindf.sold])
    xtrain_curr = [CState(x) for x in traindf.scurr]
    vωindices = [9, 10, 22, 23, 35, 36, 48, 49, 11, 24, 37, 50]  # v12, v13, v22, v23, v32, v33, v42, v43, ω11, ω21, ω31, ω41
    ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]
    xtest_old = [CState(x) for x in testdf.sold]
    xtest_future = [CState(x) for x in testdf.sfuture]

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1., (10 ./(stdx))...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{CState{Float64,4}}()
    gps = Vector{GPE}()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    getvω(μ) = return [SVector(0,μ[1:2]...), SVector(0,μ[3:4]...), SVector(0,μ[5:6]...), SVector(0,μ[7:8]...)], [SVector(μ[9],0,0), SVector(μ[10],0,0), SVector(μ[11],0,0), SVector(μ[12],0,0)]
    for i in 1:length(xtest_old)
        predictedstate, _ = predictdynamics(mechanism, gps, xtest_old[i], config["simsteps"], getvω; regularizer = 1e-10)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future, params
end
