using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics
using JSON


function experimentNoisyCPMinSin(config)
    Σ = config["Σ"]
    ΔJ = [SMatrix{3,3,Float64}(Σ["J"]randn(9)...), SMatrix{3,3,Float64}(Σ["J"]randn(9)...)]
    m = abs.(ones(2) .+ Σ["m"]randn(2))
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()-0.5)π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exp2 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()/2+0.5)*rand([-1,1])π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exptest = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=2π*(rand()-0.5), vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = cartpole(1, Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    l = mechanism.bodies[2].shape.rh[2]

    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "CP", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [tocstate(x) for x in traindf.sold]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_old = [[s[1], s[2], sin(s[3]), s[4]] for s in xtrain_old]
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # v12, ω21
    xtest_old = [tocstate(x) for x in testdf.sold]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector{GPE}()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "CP", gps, xtest_old[i], config["simsteps"]; usesin=true)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end
