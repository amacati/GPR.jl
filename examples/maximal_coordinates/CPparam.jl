using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentCPMax(config, _)
    mechanism = deepcopy(config["mechanism"])
    # Sample from dataset
    xtrain_old = reduce(hcat, [CState(x) for x in config["traindf"].sold])
    xtrain_curr = [CState(x) for x in config["traindf"].scurr]
    vωindices = [9, 22, 23, 24]
    ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]
    xtest_old = [CState(x) for x in config["testdf"].sold]
    xtest_future = [CState(x) for x in config["testdf"].sfuture]

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
