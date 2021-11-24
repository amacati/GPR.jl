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
    friction = rand(2) .* [4., 4.]
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=friction, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = fourbar(1; Δt=0.01, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.xyz[3]
    
    xtest_future_true = deepcopy([tocstate(s) for s in testdf.sfuture])
    # Add noise to the dataset
    for df in [traindf, testdf]
        applynoise!(df, Σ, "FB", config["Δtsim"], l)
    end
    # Create train and testsets
    xtrain_old = [tocstate(s) for s in traindf.sold]
    xtrain_old = [max2mincoordinates_fb(cstate) for cstate in xtrain_old]
    xtrain_old = [[sin(s[1]), s[2], sin(s[3]), s[4]] for s in xtrain_old]
    xtrain_old = reduce(hcat, xtrain_old)
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_curr = [max2mincoordinates_fb(cstate) for cstate in xtrain_curr]
    vωindices = [11, 37]
    ytrain = [[s[id] for s in xtrain_curr] for id in [2,4]]  # ω11, ω31
    xtest_old = [tocstate(s) for s in testdf.sold]
    xtest_old = [max2mincoordinates_fb(cstate) for cstate in xtest_old]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"] .* 1e-10
    gps = Vector{GPE}()
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
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanDynamics(mechanism, getμ(vωindices[id]), xtransform=xtransform)
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    for i in 1:length(xtest_old)
        predictedstate = predictdynamicsmin(mechanism, "FB", gps, xtest_old[i], config["simsteps"], usesin = true)
        push!(predictedstates, predictedstate)
    end
    return predictedstates, xtest_future_true
end
