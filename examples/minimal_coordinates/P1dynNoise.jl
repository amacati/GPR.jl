using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentMeanDynamicsNoisyP1Min(config)
    Σ = config["Σ"]
    ΔJ = SMatrix{3,3,Float64}(Σ["J"]randn(9)...)
    m = abs.(1. .+ Σ["m"]randn())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    exp1 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5) * π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    exp2 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=((rand()/2 + 0.5)*rand((-1,1))) * π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]  # [-π:-π/2; π/2:π] [-π:π]
    exptest = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5)*2π, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[1]
    traindf, testdf = generate_dataframes(config, config["nsamples"], exp1, exp2, exptest)
    mechanism = simplependulum2D(1, Δt=0.01, m = m, ΔJ = ΔJ, threadlock = config["mechanismlock"])[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M
    l = mechanism.bodies[1].shape.rh[2]
    xtest_curr_true = deepcopy([tocstate(x) for x in testdf.scurr])  # Without noise
    xtest_curr_true = [max2mincoordinates(cstate, mechanism) for cstate in xtest_curr_true]
    xtest_future_true = deepcopy([tocstate(x) for x in testdf.sfuture])

    # Add noise to the dataset
    for df in [traindf, testdf]
        for col in eachcol(df)
            for t in 1:length(col)
                col[t][1].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][1].qc  # Small error around θ
                col[t][1].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
                θ = Rotations.rotation_angle(col[t][1].qc)*sign(col[t][1].qc.x)*sign(col[t][1].qc.w)  # Signum for axis direction
                ω = col[t][1].ωc[1]
                col[t][1].xc = [0, l/2*sin(θ), -l/2*cos(θ)]  # Noise is consequence of θ and ω
                col[t][1].vc = [0, ω*cos(θ)*l/2, ω*sin(θ)*l/2]  # l/2 because measurement is in the center of the pendulum
            end
        end
    end

    # Create train and testsets
    xtrain_old = [tocstate(x) for x in traindf.sold]
    xtrain_curr = [tocstate(x) for x in traindf.scurr]
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    ytrain = [s[2] for s in xtrain_curr]
    xtest_old = [tocstate(x) for x in testdf.sold]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]

    predictedstates = Vector{Vector{Float64}}()
    params = config["params"]
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(xtrain_old, ytrain, MeanDynamics(mechanism, 1, 4, coords = "min"), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
    
    for i in 1:length(xtest_old)
        θold, ωold = xtest_old[i]
        θcurr, _ = xtest_curr_true[i]
        for _ in 1:config["simsteps"]
            ωcurr = predict_y(gp, reshape([θold, ωold], :, 1))[1][1]
            θold, ωold = θcurr, ωcurr
            θcurr = θcurr + ωcurr*mechanism.Δt
        end
        q1 = UnitQuaternion(RotX(θcurr))
        vq1 = [q1.w, q1.x, q1.y, q1.z]
        cstate = [0, 0.5l*sin(θcurr), -0.5l*cos(θcurr), vq1..., zeros(6)...]
        push!(predictedstates, cstate)
    end
    return predictedstates, xtest_future_true
end

