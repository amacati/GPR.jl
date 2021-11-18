using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyCPMinSin(config)
    Σ = config["Σ"]
    ΔJ = [SMatrix{3,3,Float64}(Σ["J"]randn(9)...), SMatrix{3,3,Float64}(Σ["J"]randn(9)...)]
    m = abs.(ones(2) .+ Σ["m"]randn(2))
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()-0.5)π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exp2 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()/2+0.5)*rand([-1,1])π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exptest = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=2π*(rand()-0.5), vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = cartpole(1, Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    l = mechanism.bodies[2].shape.rh[2]
    xtest_curr_true = deepcopy([tocstate(x) for x in testdf.scurr])  # Without noise
    xtest_curr_true = [max2mincoordinates(cstate, mechanism) for cstate in xtest_curr_true]  # Noise free
    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])

    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "CP", config["Δtsim"], l)
    end
    
    # Create train and testsets
    xtrain_old = [tocstate(x) for x in traindf.sold]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_old = [[s[1], s[2], sin(s[3]), s[4]] for s in xtrain_old]  # Convert to sin(θ)
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]  # Don't convert, only used for velocities
    xtrain_old = reduce(hcat, xtrain_old)
    yv = [s[2] for s in xtrain_curr]
    yω = [s[4] for s in xtrain_curr]
    ytrain = [yv, yω]
    xtest_old = [tocstate(x) for x in testdf.sold]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]  # Don't convert, used as is for angle computation

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    function tfmin(x, mechanism)
        return min2maxcoordinates([x[1], x[2], asin(x[3]), x[4]], mechanism)
    end
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        id == 1 ? entryID = 2 : entryID = 4
        mean = MeanDynamics(mechanism, id, entryID, coords = "min", tfmin = tfmin)
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_old)
        xold, vold, θold, ωold = xtest_old[i]
        xcurr, _, θcurr, _ = xtest_curr_true[i]
        for _ in 1:config["simsteps"]
            vcurr, ωcurr = predict_velocities(gps, reshape([xold, vold, sin(θold), ωold], :, 1))
            xold, vold, θold, ωold = xcurr, vcurr, θcurr, ωcurr
            xcurr = xcurr + vcurr*mechanism.Δt
            θcurr = θcurr + ωcurr*mechanism.Δt
        end
        q = UnitQuaternion(RotX(θcurr))
        vq = [q.w, q.x, q.y, q.z]
        cstate = [0, xcurr, 0, 1, zeros(10)..., 0.5l*sin(θcurr)+xcurr, -0.5l*cos(θcurr), vq..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_future_true
end
