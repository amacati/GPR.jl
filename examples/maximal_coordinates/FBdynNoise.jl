using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyFBMax(config)
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1 .+ Σ["m"]randn())
    friction = rand(2) .* [4., 4.]
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = fourbar(1; Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    
    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "FB", config["Δtsim"], mechanism.bodies[1].shape.xyz[3])
    end
    # Create train and testsets
    xtrain_old = reduce(hcat, [tocstate(x) for x in traindf.sold])
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    vωindices = [9, 10, 22, 23, 35, 36, 48, 49, 11, 24, 37, 50]  # v12, v13, v22, v23, v32, v33, v42, v43, ω11, ω21, ω31, ω41
    ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]
    xtest_old = [tocstate(x) for x in testdf.sold]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector{GPE}()
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, getμ(vωindices[id]))
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    projectionerror = 0
    getvω(μ) = return [SVector(0,μ[1:2]...), SVector(0,μ[3:4]...), SVector(0,μ[5:6]...), SVector(0,μ[7:8]...)], [SVector(μ[9],0,0), SVector(μ[10],0,0), SVector(μ[11],0,0), SVector(μ[12],0,0)]
    for i in 1:length(xtest_old)
        predictedstate, perror = predictdynamics(mechanism, gps, xtest_old[i], config["simsteps"], getvω; regularizer = 1e-10)
        push!(predictedstates, predictedstate)
        projectionerror += perror
    end
    return predictedstates, xtest_future_true, projectionerror/length(xtest_old)
end
