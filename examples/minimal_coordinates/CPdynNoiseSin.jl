using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyCPMinSin(config, id)
    traindfs, testdfs = loaddatasets("CPfriction")
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["nsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]
    mechanism = cartpole(1, Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    l = mechanism.bodies[2].shape.rh[2]

    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, config["Σ"], "CP", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [max2mincoordinates(CState(x), mechanism) for x in traindf.sold]
    xtrain_old = reduce(hcat, [[s[1], s[2], sin(s[3]), s[4]] for s in xtrain_old])  # Convert to sin(θ)
    xtrain_curr = [max2mincoordinates(CState(x), mechanism) for x in traindf.scurr]
    vωindices = [9, 24]
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # v12, ω21
    xtest_old = [max2mincoordinates(CState(x),mechanism) for x in testdf.sold]

    predictedstates = Vector{CState{Float64,2}}()
    params = config["params"]
    gps = Vector{GPE}()
    function xtransform(x, mech)
        x, v, θ, ω = x[1], x[2], asin(x[3]), x[4]
        q = UnitQuaternion(RotX(θ))
        θ = Rotations.rotation_angle(q)*sign(q.x)*sign(q.w)  # Signum for axis direction
        x2curr = [0, x, 0] + [0, .5sin(θ)l, -.5cos(θ)l]
        θnext = ω*0.01 + θ
        x2next = [0, x, 0] + [0, v, 0].*0.01 + [0, .5sin(θnext)l, -.5cos(θnext)l]
        v2 = (x2next - x2curr)/0.01
        return [0, x, 0, 1, 0, 0, 0, 0, v, 0, 0, 0, 0,
                x2curr..., q.w, q.x, q.y, q.z, v2..., ω, 0, 0]
    end
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, getμ(vωindices[id]), xtransform=xtransform)
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "CP", gps, xtest_old[i], config["simsteps"]; usesin = true)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end
