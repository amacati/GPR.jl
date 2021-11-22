using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyP1Max(config)
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs(1. + Σ["m"]randn())
    friction = rand()
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5) * π, ωstart = 2(rand()-0.5), m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exp2 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=((rand()/2 + 0.5)*rand((-1,1))) * π, ωstart = 2(rand()-0.5), m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]  # [-π:-π/2; π/2:π] [-π:π]
    exptest = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5)*2π, ωstart = 2(rand()-0.5), m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = simplependulum2D(1, Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    xtest_old_true = deepcopy([tocstate(x) for x in testdf.sold])  # Without noise
    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])

    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "P1", config["Δtsim"], mechanism.bodies[1].shape.rh[2])
    end

    # Create train and testsets
    xtrain_old = [tocstate(x) for x in traindf.sold]
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_old = reduce(hcat, xtrain_old)
    yv2 = [s[9] for s in xtrain_curr]
    yv3 = [s[10] for s in xtrain_curr]
    yω = [s[11] for s in xtrain_curr]
    ytrain = [yv2, yv3, yω]
    xtest_old = [tocstate(x) for x in testdf.sold]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    getμ1(mech) = mech.bodies[1].state.vsol[2][2]
    getμ2(mech) = mech.bodies[1].state.vsol[2][3]
    getμ3(mech) = mech.bodies[1].state.ωsol[2][1]
    getμs = [getμ1, getμ2, getμ3]
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

    projectionerror = 0
    for i in 1:length(xtest_old)
        setstates!(mechanism, tovstate(xtest_old_true[i]))
        oldstates = xtest_old[i]
        for _ in 1:config["simsteps"]
            μ = predict_velocities(gps, reshape(oldstates, :, 1))
            vcurr, ωcurr = [SVector(0, μ[1:2]...)], [SVector(μ[3], 0, 0)]
            vconst, ωconst = projectv!(vcurr, ωcurr, mechanism)
            projectionerror += norm(vcat(reduce(vcat, vconst .- vcurr), reduce(vcat, ωconst .-ωcurr)))
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_future_true, projectionerror/(length(xtest_old)*config["simsteps"])
end
