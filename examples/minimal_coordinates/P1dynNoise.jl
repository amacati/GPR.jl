using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyP1Min(config)
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1. .+ Σ["m"]randn())
    friction = rand()
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5) * π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exp2 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=((rand()/2 + 0.5)*rand((-1,1))) * π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]  # [-π:-π/2; π/2:π] [-π:π]
    exptest = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5)*2π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = simplependulum2D(1, Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.rh[2]

    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "P1", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [tocstate(x) for x in traindf.sold]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_old = reduce(hcat, xtrain_old)
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    ytrain = [s[2] for s in xtrain_curr]
    xtest_old = [tocstate(x) for x in testdf.sold]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    function xtransform(x, _)
        θ, ω = x[1], x[2]
        q = UnitQuaternion(RotX(θ))
        xcurr = [0, .5sin(θ)l, -.5cos(θ)l]
        θnext = θ + ω*0.01
        xnext = [0, .5sin(θnext)l, -.5cos(θnext)l]
        v = (xnext - xcurr) / 0.01
        return [xcurr..., q.w, q.x, q.y, q.z, v..., ω, 0, 0]
    end
    gp = GP(xtrain_old, ytrain, MeanDynamics(mechanism, getμ(11), xtransform=xtransform), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
    gps = [gp]

    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "P1", gps, xtest_old[i], config["simsteps"])
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end
