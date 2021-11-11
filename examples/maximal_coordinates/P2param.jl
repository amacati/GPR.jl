using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics


function experimentP2Max(config)
    mechanism = deepcopy(config["mechanism"])
    # Sample from dataset
    dataset = config["dataset"]
    testsets = StatsBase.sample(1:length(dataset.storages), config["ntestsets"], replace=false)
    trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
    xtrain_t0, xtrain_t1 = sampledataset(dataset, config["nsamples"], Δt = config["Δtsim"], random = true, exclude = testsets, stepsahead = 0:1)
    xtrain_t0 = reduce(hcat, xtrain_t0)
    yv12 = [s[9] for s in xtrain_t1]
    yv13 = [s[10] for s in xtrain_t1]
    yv22 = [s[22] for s in xtrain_t1]
    yv23 = [s[23] for s in xtrain_t1]
    yω11 = [s[11] for s in xtrain_t1]
    yω21 = [s[24] for s in xtrain_t1]
    ytrain = [yv12, yv13, yv22, yv23, yω11, yω21]
    xtest_t0, xtest_tk = sampledataset(dataset, config["testsamples"], Δt = config["Δtsim"], random = true, 
                                       pseudorandom = true, exclude = trainsets, stepsahead=[0,config["simsteps"]+1])
    # Sample random parameters
    stdx = std(xtrain_t0, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1.1, (50 ./stdx)...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_t0, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_t0)
        setstates!(mechanism, tovstate(xtest_t0[i]))
        oldstates = xtest_t0[i]
        for _ in 1:config["simsteps"]
            μ = predict_velocities(gps, reshape(reduce(vcat, oldstates), :, 1))
            vcurr, ωcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...)], [SVector(μ[5], 0, 0), SVector(μ[6], 0, 0)]
            projectv!(vcurr, ωcurr, mechanism)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
            oldstates = getcstate(mechanism)
        end
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
        push!(predictedstates, getcstate(mechanism))  # Extract xnew, write as result
    end
    return predictedstates, xtest_tk, params
end

function simulation(config, params)
    mechanism = deepcopy(config["mechanism"])
    storage = Storage{Float64}(300, length(mechanism.bodies))
    initialstates = tovstate(config["x_test"][1])
    for id in 1:length(mechanism.bodies)
        storage.x[id][1] = initialstates[id][1:3]
        storage.q[id][1] = UnitQuaternion(initialstates[id][4], initialstates[id][5:7])
        storage.v[id][1] = initialstates[id][8:10]
        storage.ω[id][1] = initialstates[id][11:13]
    end

    gps = Vector()
    for yi in config["y_train"]
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(config["x_train"], yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end

    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    states = tovstate(config["x_test"][1])
    setstates!(mechanism, states)
    for i in 2:length(storage.x[1])
        μ = predict_velocities(gps, reshape(reduce(vcat, states), :, 1))
        vcurr, ωcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...)], [SVector(μ[5], 0, 0), SVector(μ[6], 0, 0)]
        projectv!(vcurr, ωcurr, mechanism, ϵ=1e-9)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getvstates(mechanism)
        overwritestorage(storage, states, i)
    end
    return storage
end

#=
include(joinpath("..", "dataset.jl"))
include(joinpath("..", "generatedata.jl"))

params = [3.940176404112413,0.05279807477641508,789.4711055827742,2979.525631065898,3709.05683102703,319.11299086931683,0.022562606791638025,0.19782183917358442,0.1661918499497776,292.092469982139,313.3312502548224,106.77438695611079,0.18783829308830874,0.12933532848470064,0.07361660789865422,115.60970222972136,33.609497193138964,1957.6347048390799,379.72881473894006,0.0524239281041154,0.17017151026550986,0.2010528903058501,64.68424546414947,179.7892313177993,19.778897167207937,0.14626717543831935,0.13022341795728232]

dataset = Dataset()
mechanism = nothing
for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
    storage, mechanism, _ = doublependulum2D(Δt=0.01, θstart=[θ1, θ2])
    dataset += storage
end
testsets = [2]
trainsets = [i for i in 1:length(dataset.storages) if !(i in testsets)]
xtrain_t0, xtrain_t1 = sampledataset(dataset, 250, Δt = 0.01, exclude = testsets, stepsahead = 0:1)
xtrain_t0 = reduce(hcat, xtrain_t0)
yv12 = [s[9] for s in xtrain_t1]
yv13 = [s[10] for s in xtrain_t1]
yv22 = [s[22] for s in xtrain_t1]
yv23 = [s[23] for s in xtrain_t1]
yω11 = [s[11] for s in xtrain_t1]
yω21 = [s[24] for s in xtrain_t1]
ytrain = [yv12, yv13, yv22, yv23, yω11, yω21]
xtest_t0 = sampledataset(dataset, 5, Δt = 0.01, exclude = trainsets, stepsahead=[0])
config = Dict("mechanism" => mechanism, "x_train" => xtrain_t0, "y_train" => ytrain, "x_test" => xtest_t0)

storage = simulation(config, params)
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
=#