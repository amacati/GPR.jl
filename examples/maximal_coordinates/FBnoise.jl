using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentNoisyFBMax(config)
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1 .+ Σ["m"]randn())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = fourbar(1; Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    xtest_old_true = deepcopy([tocstate(x) for x in testdf.sold])  # Without noise
    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])

    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "FB", config["Δtsim"], mechanism.bodies[1].shape.xyz[3])
    end

    # Create train and testsets
    xtrain_old = [tocstate(x) for x in traindf.sold]
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_old = reduce(hcat, xtrain_old)
    yv12 = [s[9] for s in xtrain_curr]
    yv13 = [s[10] for s in xtrain_curr]
    yv22 = [s[22] for s in xtrain_curr]
    yv23 = [s[23] for s in xtrain_curr]
    yv32 = [s[35] for s in xtrain_curr]
    yv33 = [s[36] for s in xtrain_curr]
    yv42 = [s[48] for s in xtrain_curr]
    yv43 = [s[49] for s in xtrain_curr]
    yω11 = [s[11] for s in xtrain_curr]
    yω21 = [s[24] for s in xtrain_curr]
    yω31 = [s[37] for s in xtrain_curr]
    yω41 = [s[50] for s in xtrain_curr]
    ytrain = [yv12, yv13, yv22, yv23, yv32, yv33, yv42, yv43, yω11, yω21, yω31, yω41]
    xtest_old = [tocstate(x) for x in testdf.sold]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
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
            μ = predict_velocities(gps, reshape(oldstates, :, 1))  # Noisy
            vcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...), SVector(0, μ[5:6]...), SVector(0, μ[7:8]...)]
            ωcurr = [SVector(μ[9], 0, 0), SVector(μ[10], 0, 0), SVector(μ[11], 0, 0), SVector(μ[12], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism, regularizer=1e-10)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_future_true
end
