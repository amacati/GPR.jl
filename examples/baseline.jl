using GPR

function experimentVarInt(config, id, eid, mechanism, varargs...)
    testdfs = config["datasets"][2]  # Each thread operates on its own dataset -> no races
    testdf = testdfs.df[id][shuffle(1:nrow(testdfs.df[id]))[1:config["testsamples"]], :]
    xtest_future_true = [CState(x) for x in testdf.sfuture]
    # Add noise to the dataset
    applynoise!(testdf, config["Σ"], eid, config["Δtsim"], varargs...)
    xtest_old = [CState(x) for x in testdf.sold]
    predictedstates = Vector{CState{Float64, length(mechanism.bodies)}}()
    for i in 1:length(xtest_old)
        predictedstate = predictdynamics(mechanism, xtest_old[i], config["simsteps"])
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end

function experimentVarIntP1(config, id)
    mechanism = simplependulum2D(1, Δt=0.01, threadlock = config["mechanismlock"])[2]
    return experimentVarInt(config, id, "P1", mechanism, mechanism.bodies[1].shape.rh[2])
end

function experimentVarIntP2(config, id)
    mechanism = doublependulum2D(1, Δt=0.01, threadlock = config["mechanismlock"])[2]
    return experimentVarInt(config, id, "P2", mechanism, mechanism.bodies[1].shape.xyz[3], mechanism.bodies[2].shape.xyz[3])
end

function experimentVarIntCP(config, id)
    mechanism = cartpole(1, Δt=0.01, threadlock = config["mechanismlock"])[2]
    return experimentVarInt(config, id, "CP", mechanism, mechanism.bodies[2].shape.rh[2])
end

function experimentVarIntFB(config, id)
    mechanism = fourbar(1, Δt=0.01, threadlock = config["mechanismlock"])[2]
    return experimentVarInt(config, id, "FB", mechanism, mechanism.bodies[1].shape.xyz[3])
end