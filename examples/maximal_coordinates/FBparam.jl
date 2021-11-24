using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentFBMax(config)
    mechanism = deepcopy(config["mechanism"])
    # Sample from dataset
    xtrain_old = reduce(hcat, [tocstate(x) for x in config["traindf"].sold])
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]
    vωindices = [9, 10, 22, 23, 35, 36, 48, 49, 11, 24, 37, 50]  # v12, v13, v22, v23, v32, v33, v42, v43, ω11, ω21, ω31, ω41
    ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1., (10 ./(stdx))...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
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
