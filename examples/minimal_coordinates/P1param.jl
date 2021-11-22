using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentP1Min(config)
    mechanism = deepcopy(config["mechanism"])
    l = mechanism.bodies[1].shape.rh[2]
    # Sample from dataset
    xtrain_old = [tocstate(x) for x in config["traindf"].sold]
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]    
    xtrain_old = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_old]
    xtrain_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtrain_curr]
    xtrain_old = reduce(hcat, xtrain_old)
    ytrain = [s[2] for s in xtrain_curr]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_curr = [tocstate(x) for x in config["testdf"].scurr]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]
    xtest_old = [max2mincoordinates(cstate, mechanism) for cstate in xtest_old]
    xtest_curr = [max2mincoordinates(cstate, mechanism) for cstate in xtest_curr]
    # intentionally not converting xtest_future since final comparison is done in maximal coordinates

    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 100
    params = [1.1, (10 ./(stdx))...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    gp = GP(xtrain_old, ytrain, MeanZero(), kernel)
    GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))

    for i in 1:length(xtest_old)
        θold, ωold = xtest_old[i]
        θcurr, _ = xtest_curr[i]
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
    return predictedstates, xtest_future, params
end

# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")