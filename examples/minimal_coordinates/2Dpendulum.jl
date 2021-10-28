using GPR
using ConstrainedDynamicsVis
using ConstrainedDynamics
using Plots
using StatsBase: sample

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


EXPERIMENT_ID = "P1_2D_MIN_GGK"
load_checkpoint = false
params_vec = collect(Iterators.product(0.1:0.5:15.1, 0.1:0.5:15.1, 0.1:0.5:15.1))
nprocessed = 0
tstart = time()
paramlock = ReentrantLock()
errorlock = ReentrantLock()

storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
data = [SVector(state[1], state[2]) for state in data]
cleardata!(data)
X = data[1:end-1]
Yω = [s[2] for s in data[2:end]]


onestep_msevec = Vector{Float64}()
onestep_params = Vector{Vector{Float64}}()
# Load checkpoints if any checkpoints are present
success, checkpointdict = loadcheckpoint(EXPERIMENT_ID)
if success && load_checkpoint
    nprocessed = checkpointdict["nprocessed"]
    onestep_msevec = checkpointdict["onestep_msevec"]
    onestep_params = checkpointdict["onestep_params"]
end

Threads.@threads for _ in nprocessed+1:length(params_vec)
    # Get hyperparameters (threadsafe)
    params = []
    lock(paramlock)
    try
        global nprocessed
        @assert nprocessed < length(params_vec)
        params = params_vec[nprocessed+1]
        nprocessed += 1
        if nprocessed % 10 == 0
            println("Processing job $nprocessed/$(length(params_vec))")
            Δt = time() - tstart
            secs = Int(round(Δt*(length(params_vec)/nprocessed-1)))
            hours = div(secs, 3600)
            minutes = div(secs-hours*3600, 60)
            secs -= (hours*3600 + minutes * 60)
            println("Estimated time to completion: $(hours)h, $(minutes)m, $(secs)s")
            if nprocessed % 100 == 0
                checkpointdict = Dict("nprocessed" => nprocessed, "onestep_msevec" => onestep_msevec, "onestep_params" => onestep_params)
                checkpoint(EXPERIMENT_ID, checkpointdict)
            end
        end
    catch e
        display(e)
        continue
    finally
        unlock(paramlock)
    end
    # Main experiment
    onestep_mse = Inf
    try
        kernel = GeneralGaussianKernel(params[1], [params[2:3]...])
        gprω = GaussianProcessRegressor(X, Yω, kernel)
        optimize!(gprω)

        resetMechanism!(mechanism, initialstates)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
        maxstates = getstates(mechanism)
        states = max2mincoordinates(maxstates, mechanism)[1]
        θ, ω = states[1], states[2]
        for i in 2:length(storage.x[1])
            ω = GPR.predict(gprω, SVector(θ, ω))[1][1]
            θ += ω*mechanism.Δt
            storage.x[1][i] = [0, 0.5sin(θ), -0.5cos(θ)]
            storage.q[1][i] = UnitQuaternion(RotX(θ))
        end
        onestep_mse = onesteperror(mechanism, storage)
        isnan(onestep_mse) ? onestep_mse = Inf : nothing
    catch e
        # display(e)
    end
    lock(errorlock)
    try
        push!(onestep_msevec, onestep_mse)
        push!(onestep_params, [params...])
    catch e
        display(e)
    finally
        unlock(errorlock)
    end
end

checkpointdict = Dict("nprocessed" => nprocessed, "onestep_msevec" => onestep_msevec, "onestep_params" => onestep_params)
checkpoint(EXPERIMENT_ID*"_FINAL", checkpointdict)

println("Best one step mean squared error: $(minimum(onestep_msevec))")
#plt = plot(1:length(onestep_msevec), onestep_msevec, yaxis=:log, seriestype = :scatter, title="Pendulum position one step MSE ", label = "pos MSE")
#xlabel!("Trial")
#savefig(plt, "2Dpendulum_min_GeneralGaussianKernel")
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
