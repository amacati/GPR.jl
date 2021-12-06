using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function _experiment_fb_min(config, id; usesin = false, meandynamics = false)
    traindfs, testdfs = config["datasets"]  # Each thread operates on its own dataset -> no races
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["nsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]    
    mechanism = fourbar(1; Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    l = mechanism.bodies[1].shape.xyz[3]

    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, config["Σ"], "FB", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [max2mincoordinates_fb(CState(x)) for x in traindf.sold]
    if usesin
        xtrain_old = reduce(hcat, [[sin(s[1]), s[2], sin(s[3]), s[4]] for s in xtrain_old])
    else
        xtrain_old = reduce(hcat, xtrain_old)
    end
    xtrain_curr = [max2mincoordinates_fb(CState(x)) for x in traindf.scurr]
    vωindices = [11, 37]
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # ω11, ω31
    xtest_old = [max2mincoordinates_fb(CState(x)) for x in testdf.sold]

    predictedstates = Vector{CState{Float64,4}}()
    params = config["params"]
    gps = Vector{GPE}()

    function xtransform(x, _)
        θ1, ω1, θ2, ω2 = x
        if usesin θ1, θ2 = asin(θ1), asin(θ2) end
        x1 = [0, .5sin(θ1)l, -.5cos(θ1)l]
        x2 = [0, sin(θ1)l + .5sin(θ2)l, -cos(θ1)l - .5cos(θ2)l]
        x3 = [0, .5sin(θ2)l, -.5cos(θ2)l]
        x4 = [0, sin(θ2)l + 0.5sin(θ1)l, -cos(θ2)l - .5cos(θ1)l]
        q1 = UnitQuaternion(RotX(θ1))
        qv1 = [q1.w, q1.x, q1.y, q1.z]
        q2 = UnitQuaternion(RotX(θ2))
        qv2 = [q2.w, q2.x, q2.y, q2.z]
        qv3 = qv2
        qv4 = qv1
        θ1next = 0.01ω1 + θ1  # 0.01 = Δt for mechanism in GP prediction
        θ2next = 0.01ω2 + θ2
        x1next = [0, .5sin(θ1next)l, -.5cos(θ1next)l]
        x2next = [0, sin(θ1next)l + .5sin(θ2next)l, -cos(θ1next)l - .5cos(θ2next)l]
        x3next = [0, .5sin(θ2next)l, -.5cos(θ2next)l]
        x4next = [0, sin(θ2next)l + .5sin(θ1next)l, -cos(θ2next)l - .5cos(θ1next)l]
        v1 = (x1next - x1)/0.01
        v2 = (x2next - x2)/0.01
        v3 = (x3next - x3)/0.01
        v4 = (x4next - x4)/0.01
        ω1 = [ω1, 0, 0]
        ω3 = [ω2, 0, 0]
        ω2 = ω3
        ω4 = ω1
        cstate = [x1..., qv1..., v1..., ω1..., x2..., qv2..., v2..., ω2...,
                  x3..., qv3..., v3..., ω3..., x4..., qv4..., v4..., ω4...]
        return cstate
    end

    cache = MDCache()
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = meandynamics ? MeanDynamics(mechanism, getμ(vωindices), id, cache, xtransform=xtransform) : MeanZero()
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "FB", gps, xtest_old[i], config["simsteps"]; usesin = usesin)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end

function experiment_fb_mz_min(config, id)
    return _experiment_fb_min(config, id, usesin=false, meandynamics=false)
end

function experiment_fb_mz_min_sin(config, id)
    return _experiment_fb_min(config, id, usesin=true, meandynamics=false)
end

function experiment_fb_md_min(config, id)
    return _experiment_fb_min(config, id, usesin=false, meandynamics=true)
end

function experiment_fb_md_min_sin(config, id)
    return _experiment_fb_min(config, id, usesin=true, meandynamics=true)
end