using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyFBMinSin(config)
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1 .+ Σ["m"]randn())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = fourbar(1; Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.xyz[3]
    xtest_curr_true = deepcopy([tocstate(s) for s in testdf.scurr])  # Without noise
    xtest_curr_true = [max2mincoordinates(cstate, mechanism) for cstate in xtest_curr_true]
    xtest_curr_true = [[s[1:2]..., s[1]+s[5], s[2]+s[6]] for s in xtest_curr_true]
    xtest_future_true = deepcopy([tocstate(s) for s in testdf.sfuture])

    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "FB", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [tocstate(s) for s in traindf.sold]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_old = [[s[1:2]..., s[1]+s[5], s[2]+s[6]] for s in xtrain_old]
    xtrain_old = [[sin(s[1]), s[2], sin(s[3]), s[4]] for s in xtrain_old]
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_curr = [[s[1:2]..., s[1]+s[5], s[2]+s[6]] for s in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    ω1 = [s[2] for s in xtrain_curr]
    ω2 = [s[4] for s in xtrain_curr]
    ytrain = [ω1, ω2]
    xtest_old = [tocstate(s) for s in testdf.sold]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]
    xtest_old = [[s[1:2]..., s[1]+s[5], s[2]+s[6]] for s in xtest_old]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    gps = Vector()
    function xtransform(x, _)
        θ1, ω1, θ2, ω2 = asin(x[1]), x[2], asin(x[3]), x[4]
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
        θ1next = 0.01ω1 + θ1
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
    getμ1(mech) = return mech.bodies[1].state.ωsol[2][1]
    getμ2(mech) = return mech.bodies[3].state.ωsol[2][1] + mech.bodies[1].state.ωsol[2][1]
    getμs = [getμ1, getμ2]
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        id == 1 ? getμ = getμ1 : getμ = getμ2
        mean = MeanDynamics(mechanism, getμs[id], xtransform=xtransform)
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_old)
        θ1old, ω1old, θ2old, ω2old = xtest_old[i]
        θ1curr, _, θ2curr, _ = xtest_curr_true[i]
        for _ in 1:config["simsteps"]
            ω1curr, ω2curr = predict_velocities(gps, reshape([sin(θ1old), ω1old, sin(θ2old), ω2old], :, 1))
            θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
            θ1curr = θ1curr + ω1curr*mechanism.Δt
            θ2curr = θ2curr + ω2curr*mechanism.Δt
        end
        x1 = [0, 0.5sin(θ1curr), -0.5cos(θ1curr)]
        x2 = [0, sin(θ1curr) + 0.5sin(θ2curr), -cos(θ1curr) - 0.5cos(θ2curr)]
        x3 = [0, 0.5sin(θ2curr), -0.5cos(θ2curr)]
        x4 = [0, sin(θ2curr) + 0.5sin(θ1curr), -cos(θ2curr) - 0.5cos(θ1curr)]
        cstate = [x1..., zeros(10)..., x2..., zeros(10)..., x3..., zeros(10)..., x4..., zeros(10)...]  # Orientation, velocities not used in error
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_future_true
end




include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))
include(joinpath("..", "mDynamics.jl"))

params = ones(5)*1e-9
tsim = 0.001
config = Dict("Δtsim"=>tsim, "testsamples"=>1, "simsteps"=>20, "nsamples"=>2)
nsteps = 2*Int(1/tsim)  # Equivalent to 2 seconds
exp1 = () -> fourbar(nsteps, Δt=tsim)[1]
exp2 = () -> fourbar(nsteps, Δt=tsim)[1]    # [-π:-π/2; π/2:π] [-π:π]
exptest = () -> fourbar(nsteps, Δt=tsim)[1]
traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
mechanism = fourbar(1, Δt=0.01)[2]

    # Create train and testsets
    xtrain_old = [tocstate(s) for s in traindf.sold]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_old = [[s[1:2]..., s[1]+s[5], s[2]+s[6]] for s in xtrain_old]
    xtrain_old = [[sin(s[1]), s[2], sin(s[3]), s[4]] for s in xtrain_old]
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_curr = [[s[1:2]..., s[1]+s[5], s[2]+s[6]] for s in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    ω1 = [s[2] for s in xtrain_curr]
    ω2 = [s[4] for s in xtrain_curr]
    ytrain = [ω1, ω2]
    xtest_old = [tocstate(s) for s in testdf.sold]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]
    xtest_old = [[s[1:2]..., s[1]+s[5], s[2]+s[6]] for s in xtest_old]

    predictedstates = Vector{Vector{Float64}}()
    params = ones(5)*1e-5
    gps = Vector()
    function xtransform(x, _)
        θ1, ω1, θ2, ω2 = asin(x[1]), x[2], asin(x[3]), x[4]
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
        θ1next = 0.01ω1 + θ1
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
    getμ1(mech) = return mech.bodies[1].state.ωsol[2][1]
    getμ2(mech) = return mech.bodies[3].state.ωsol[2][1] + mech.bodies[1].state.ωsol[2][1]
    getμs = [getμ1, getμ2]
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, getμs[id], xtransform=xtransform)
        gp = GP(xtrain_old, yi, mean, kernel)
        # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

storage = Storage{Float64}(200, 4)
mechanism
s, mechanism = fourbar(1, Δt=0.01, θstart=[π/4, π/4])[1:2]
θ1old, ω1old, θ2old, ω2old = max2mincoordinates(getcstate(s, 1), mechanism)
θ2old, ω2old = θ2old + θ1old, ω2old + ω1old
θ1curr = θ1old + ω1old*mechanism.Δt
θ2curr = θ2old + ω2old*mechanism.Δt
for i in 1:200
    ω1curr, ω2curr = predict_velocities(gps, reshape([sin(θ1old), ω1old, sin(θ2old), ω2old], :, 1))
    display(i)
    display(ω1curr)
    θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
    θ1curr = θ1curr + ω1curr*mechanism.Δt
    θ2curr = θ2curr + ω2curr*mechanism.Δt
    q1 = UnitQuaternion(RotX(θ1curr))
    q1 = [q1.w, q1.x, q1.y, q1.z]
    q2 = UnitQuaternion(RotX(θ1curr))
    q2 = [q2.w, q2.x, q2.y, q2.z]
    x1 = [0, 0.5sin(θ1curr), -0.5cos(θ1curr)]
    x2 = [0, sin(θ1curr) + 0.5sin(θ2curr), -cos(θ1curr) - 0.5cos(θ2curr)]
    x3 = [0, 0.5sin(θ2curr), -0.5cos(θ2curr)]
    x4 = [0, sin(θ2curr) + 0.5sin(θ1curr), -cos(θ2curr) - 0.5cos(θ1curr)]
    cstate = [x1..., q1..., zeros(6)..., x2..., q2..., zeros(6)..., x3..., q2..., zeros(6)..., x4..., q1..., zeros(6)...]
    overwritestorage(storage, tovstate(cstate), i)
end
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
