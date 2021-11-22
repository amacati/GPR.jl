using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyCPMax(config)
    Σ = config["Σ"]
    ΔJ = [SMatrix{3,3,Float64}(Σ["J"]randn(9)...), SMatrix{3,3,Float64}(Σ["J"]randn(9)...)]
    m = abs.(ones(2) .+ Σ["m"]randn(2))
    friction = rand(2) .* [4., 0.3]
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()-0.5)π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exp2 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()/2+0.5)*rand([-1,1])π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exptest = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=2π*(rand()-0.5), vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = cartpole(1, Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    xtest_old_true = deepcopy([tocstate(x) for x in testdf.sold])  # Without noise
    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])

    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "CP", config["Δtsim"], mechanism.bodies[2].shape.rh[2])
    end

    # Create train and testsets
    xtrain_old = [tocstate(x) for x in traindf.sold]
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_old = reduce(hcat, xtrain_old)
    yv12 = [s[9] for s in xtrain_curr]
    yv22 = [s[22] for s in xtrain_curr]
    yv23 = [s[23] for s in xtrain_curr]
    yω21 = [s[24] for s in xtrain_curr]
    ytrain = [yv12, yv22, yv23, yω21]
    xtest_old = [tocstate(x) for x in testdf.sold]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    getμ1(mech) = return mech.bodies[1].state.vsol[2][2]
    getμ2(mech) = return mech.bodies[2].state.vsol[2][2]
    getμ3(mech) = return mech.bodies[2].state.vsol[2][3]
    getμ4(mech) = return mech.bodies[2].state.ωsol[2][1]
    getμs = [getμ1, getμ2, getμ3, getμ4]
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, getμs[id])
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_old)
        setstates!(mechanism, tovstate(xtest_old_true[i]))
        oldstates = xtest_old[i]
        for _ in 1:config["simsteps"]
            μ = predict_velocities(gps, reshape(xtest_old[i], :, 1))
            vcurr, ωcurr = [SVector(0, μ[1], 0), SVector(0, μ[2:3]...)], [SVector(zeros(3)...), SVector(μ[4], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_future_true
end
