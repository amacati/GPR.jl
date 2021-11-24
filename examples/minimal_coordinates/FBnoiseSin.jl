using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentNoisyFBMinSin(config)
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1 .+ Σ["m"]randn())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = fourbar(1; Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.xyz[3]

    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "FB", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [tocstate(s) for s in traindf.sold]
    xtrain_old = [max2mincoordinates_fb(cstate) for cstate in xtrain_old]
    xtrain_old = [[sin(s[1]), s[2], sin(s[3]), s[4]] for s in xtrain_old]
    xtrain_old = reduce(hcat, xtrain_old)
    xtrain_curr = [tocstate(s) for s in traindf.scurr]
    xtrain_curr = [max2mincoordinates_fb(cstate) for cstate in xtrain_curr]
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # ω11, ω31
    xtest_old = [tocstate(s) for s in testdf.sold]
    xtest_old = [max2mincoordinates_fb(cstate) for cstate in xtest_old]

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
        predictedstate = predictdynamicsmin(mechanism, "FB", gps, xtest_old[i], config["simsteps"], usesin = true)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end