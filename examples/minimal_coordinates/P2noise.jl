using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function _experiment_p2_min(config, id; usesin = false, meandynamics = false)
    traindfs, testdfs = config["datasets"]  # Each thread operates on its own dataset -> no races
    traindf = traindfs.df[id][shuffle(1:nrow(traindfs.df[id]))[1:config["nsamples"]], :]
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]    
    mechanism = doublependulum2D(1; Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism
    l1, l2 = mechanism.bodies[1].shape.xyz[3], mechanism.bodies[2].shape.xyz[3]

    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, config["Σ"], "P2", config["Δtsim"], l1, l2)
    end
    # Create train and testsets
    xtrain_old = [max2mincoordinates(CState(x), mechanism) for x in traindf.sold]
    if usesin
        xtrain_old = reduce(hcat, [[sin(s[1]), cos(s[1]), s[2], sin(s[3]), cos(s[3]), s[4]] for s in xtrain_old])
    else
        xtrain_old = reduce(hcat, xtrain_old)
    end
    xtrain_curr = [max2mincoordinates(CState(x), mechanism) for x in traindf.scurr]
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # ω11, ω21
    xtest_old = [max2mincoordinates(CState(x),mechanism) for x in testdf.sold]

    predictedstates = Vector{CState{Float64,2}}()
    params = config["params"]
    params = [params[1], params[2], params[2], params[3], params[4], params[4], params[5]]
    gps = Vector{GPE}()

    function xtransform(x, _)
        if usesin
            sθ1, cθ1, ω1, sθ2, cθ2, ω2 = x
            θ1, θ2 = atan(sθ1, cθ1), atan(sθ2, cθ2)
        else
            θ1, ω1, θ2, ω2 = x
        end
        q1, q2 = UnitQuaternion(RotX(θ1)), UnitQuaternion(RotX(θ1+θ2))
        x1curr = [0, .5sin(θ1)l1, -.5cos(θ1)l1]
        x2curr = [0, sin(θ1)l1 + .5sin(θ1+θ2)l2, -cos(θ1)l1 - .5cos(θ1+θ2)l2]
        θ1next = θ1 + ω1*0.01
        θ2next = θ2 + ω2*0.01
        x1next = [0, .5sin(θ1next)l1, -.5cos(θ1next)l1]
        x2next = [0, sin(θ1next)l1 + .5sin(θ1next+θ2next)l2, -cos(θ1next)l1 - .5cos(θ1next+θ2next)l2]
        v1 = (x1next - x1curr) / 0.01
        v2 = (x2next - x2curr) / 0.01
        return [x1curr..., q2vec(q1)..., v1..., ω1, 0, 0,
                x2curr..., q2vec(q2)..., v2..., ω1+ω2, 0, 0]
    end

    _getμ(mech) = return [mech.bodies[1].state.ωsol[2][1], mech.bodies[2].state.ωsol[2][1] - mech.bodies[1].state.ωsol[2][1]]
    cache = MDCache()
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = meandynamics ? MeanDynamics(mechanism, _getμ, id, cache, xtransform=xtransform) : MeanZero()
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "P2", gps, xtest_old[i], config["simsteps"]; usesin = usesin)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end

function experiment_p2_mz_min(config, id)
    return _experiment_p2_min(config, id, usesin=false, meandynamics=false)
end

function experiment_p2_mz_min_sin(config, id)
    return _experiment_p2_min(config, id, usesin=true, meandynamics=false)
end

function experiment_p2_md_min(config, id)
    return _experiment_p2_min(config, id, usesin=false, meandynamics=true)
end

function experiment_p2_md_min_sin(config, id)
    return _experiment_p2_min(config, id, usesin=true, meandynamics=true)
end