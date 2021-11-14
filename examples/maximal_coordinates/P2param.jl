using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics
using DataFrames


function experimentP2Max(config)
    mechanism = deepcopy(config["mechanism"])
    # Sample from dataset
    xtrain_old = [tocstate(x) for x in config["traindf"].sold]
    xtrain_curr = [tocstate(x) for x in config["traindf"].scurr]
    xtrain_old = reduce(hcat, xtrain_old)
    yv12 = [s[9] for s in xtrain_curr]
    yv13 = [s[10] for s in xtrain_curr]
    yv22 = [s[22] for s in xtrain_curr]
    yv23 = [s[23] for s in xtrain_curr]
    yω11 = [s[11] for s in xtrain_curr]
    yω21 = [s[24] for s in xtrain_curr]
    ytrain = [yv12, yv13, yv22, yv23, yω11, yω21]
    xtest_old = [tocstate(x) for x in config["testdf"].sold]
    xtest_future = [tocstate(x) for x in config["testdf"].sfuture]

    # Sample random parameters
    stdx = std(xtrain_old, dims=2)
    stdx[stdx .== 0] .= 1000
    params = [1.1, (50 ./stdx)...]
    params = params .+ (5rand(length(params)) .- 0.999) .* params

    predictedstates = Vector{Vector{Float64}}()
    gps = Vector()
    for yi in ytrain
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        gp = GP(xtrain_old, yi, MeanZero(), kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    function predict_velocities(gps, oldstates)
        return [predict_y(gp, oldstates)[1][1] for gp in gps]
    end

    for i in 1:length(xtest_old)
        setstates!(mechanism, tovstate(xtest_old[i]))
        oldstates = xtest_old[i]
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
    return predictedstates, xtest_future, params
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
        push!(gps, gp)
    end
    Threads.@threads for gp in gps
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
    end

    function predict_velocities(gps, states)
        return [predict_y(gp, states)[1][1] for gp in gps]
    end

    states = tovstate(config["x_test"][1])
    setstates!(mechanism, states)
    for i in 2:length(storage.x[1])
        μ = predict_velocities(gps, reshape(reduce(vcat, states), :, 1))
        vcurr, ωcurr = [SVector(0, μ[1:2]...), SVector(0, μ[3:4]...)], [SVector(μ[5], 0, 0), SVector(μ[6], 0, 0)]
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        states = getvstates(mechanism)
        overwritestorage(storage, states, i)
    end
    return storage
end

#=
include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))

params = [3.940176404112413,0.05279807477641508,789.4711055827742,2979.525631065898,3709.05683102703,319.11299086931683,0.022562606791638025,0.19782183917358442,0.1661918499497776,292.092469982139,313.3312502548224,106.77438695611079,0.18783829308830874,0.12933532848470064,0.07361660789865422,115.60970222972136,33.609497193138964,1957.6347048390799,379.72881473894006,0.0524239281041154,0.17017151026550986,0.2010528903058501,64.68424546414947,179.7892313177993,19.778897167207937,0.14626717543831935,0.13022341795728232]
t1 = time()
tsim = 0.001
traindf = DataFrame(s0 = Vector{Vector{State}}(), s1 = Vector{Vector{State}}())
scaling = Int(0.01/tsim)
nsamples = 250
ntestsamples = 1000
for _ in 1:div(nsamples, 2)
    θstart = [(rand() - 0.5), 2(rand()-0.5)] .* π
    storage = doublependulum2D(2*Int(1/tsim), Δt=tsim, θstart=θstart)[1]  # Simulate 2 secs from random position, choose one sample
    j = rand(1:2*Int(1/tsim) - (scaling))  # End of storage - required steps
    push!(traindf, (getstates(storage, j), getstates(storage, j+scaling)))
end
for _ in 1:div(nsamples, 2)
    θstart = [((rand()/2 + 0.5)*rand((-1,1))), 2(rand()-0.5)] .* π  # [-π:-π/2; π/2:π] [-π:π]
    storage = doublependulum2D(2*Int(1/tsim), Δt=tsim, θstart=θstart)[1]  # Simulate 2 secs from random position, choose one sample
    j = rand(1:2*Int(1/tsim) - (scaling))  # End of storage - required steps
    push!(traindf, (getstates(storage, j), getstates(storage, j+scaling)))
end
testdf = DataFrame(s0 = Vector{Vector{State}}(), s1 = Vector{Vector{State}}(), s21 = Vector{Vector{State}}())
for _ in 1:ntestsamples
    θstart = (rand(2) .- 0.5).*2π
    storage = doublependulum2D(2*Int(1/tsim), Δt=tsim, θstart=θstart)[1]  # Simulate 2 secs from random position, choose one sample
    j = rand(1:2*Int(1/tsim) - (scaling*21))  # End of storage - required steps
    push!(testdf, (getstates(storage, j), getstates(storage, j+scaling), getstates(storage, j+scaling*21)))
end
t2 = time()
display(t2-t1)
mechanism = doublependulum2D(1, Δt=0.01)[2]

xtrain_old = [tocstate(x) for x in traindf.s0]
xtrain_curr = [tocstate(x) for x in traindf.s1]
xtrain_old = reduce(hcat, xtrain_old)

stdx = std(xtrain_old, dims=2)
stdx[stdx .== 0] .= 1000
params = [1., (50 ./stdx)...]

yv12 = [s[9] for s in xtrain_curr]
yv13 = [s[10] for s in xtrain_curr]
yv22 = [s[22] for s in xtrain_curr]
yv23 = [s[23] for s in xtrain_curr]
yω11 = [s[11] for s in xtrain_curr]
yω21 = [s[24] for s in xtrain_curr]
ytrain = [yv12, yv13, yv22, yv23, yω11, yω21]

xtest_old = [tocstate(x) for x in testdf.s0]
config = Dict("mechanism" => mechanism, "x_train" => xtrain_old, "y_train" => ytrain, "x_test" => xtest_old)

storage = simulation(config, params)
θstart = [-3/4 * π, 1/4 * π]
ωstart = [3, -1]
# storage, mechanism, _ = doublependulum2D(200, Δt=0.01, θstart=θstart, ωstart=ωstart)
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
=#