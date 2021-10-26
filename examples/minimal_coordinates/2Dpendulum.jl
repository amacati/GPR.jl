using GPR
using ConstrainedDynamicsVis
using ConstrainedDynamics
using Plots
using StatsBase: sample


include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


params_vec = collect(Iterators.product(0.1:0.5:10.1, 0.1:0.5:10.1, 0.1:0.5:10.1))
nprocessed = 0
tstart = time()
paramlock = ReentrantLock()
errorlock = ReentrantLock()

onestep_msevec = Vector{Float64}()
onestep_params = Vector{Vector{Float64}}()

Threads.@threads for _ in 1:length(params_vec)
    # Get hyperparameters (threadsafe)
    global nprocessed
    params = []
    lock(paramlock)
    try
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
            println("Estimated time to completion: $(hours)h, $(minutes)m, $(secs)h")
        end
    catch e
        display(e)
        continue
    finally
        unlock(paramlock)
    end

    onestep_mse = Inf
    for _ in 1:5
        try
            storage, mechanism, initialstates = simplependulum2D()
            data = loaddata(storage; coordinates="minimal", mechanism = mechanism)
            data = [SVector(state[1], state[2]) for state in data]
            cleardata!(data)
            nsamples = Int(round(length(data)/5))
            sampleidx = sample(1:length(data)-1, nsamples; replace = false)

            X = data[sampleidx]
            Yω = [s[2] for s in data[sampleidx.+1]]

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
                #=ConstrainedDynamics.setVelocity!(mechanism, mechanism.eqconstraints[2], [ω])  # Pendulum only has 1 eqc
                ConstrainedDynamics.setPosition!(mechanism, mechanism.eqconstraints[2], [θ])
                foreachactive(discretizestate!, mechanism.bodies, mechanism.Δt)
                foreachactive(setsolution!, mechanism.bodies)  # Set ω as solution ωsol[2]
                foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
                maxstates = getstates(mechanism)
                overwritestorage(storage, maxstates, i)
                θ = ConstrainedDynamics.minimalCoordinates(mechanism, mechanism.eqconstraints[2])[1]
                maxstates = getstates(mechanism)
                overwritestorage(storage, maxstates, i)=#
                storage.x[1][i] = [0, 0.5sin(θ), -0.5cos(θ)]
                storage.q[1][i] = UnitQuaternion(RotX(θ))
            end

            onestep_mse = min(onestep_mse, onesteperror(mechanism, storage))
        catch e
        end
    end
    lock(errorlock)
    try
        # println("One step mean squared error: $onestep_mse")
        push!(onestep_msevec, onestep_mse)
        push!(onestep_params, [params...])
    catch e
        display(e)
    finally
        unlock(errorlock)
    end
end

println("Best one step mean squared error: $(minimum(onestep_msevec))")
plt = plot(1:length(onestep_msevec), onestep_msevec, yaxis=:log, seriestype = :scatter, title="Pendulum position one step MSE ", label = "pos MSE")
xlabel!("Trial")
savefig(plt, "2Dpendulum_min_GeneralGaussianKernel")
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")