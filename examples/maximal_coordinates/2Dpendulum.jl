using GPR
using ConstrainedDynamicsVis
using Plots
using StatsBase: sample

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage)
cleardata!(data)

nsamples = Int(round(length(data)/5))
nruns = 100

onestep_msevec = Vector{Float64}()
for _ in 1:nruns
    try
        resetMechanism!(mechanism, initialstates)  # Reset mechanism to starting position
        sampleidx = sample(1:length(data)-1, nsamples; replace = false)  # Draw random training samples

        X = data[sampleidx]
        Yv1 = [s[8] for s in data[sampleidx.+1]]
        Yv2 = [s[9] for s in data[sampleidx.+1]]
        Yv3 = [s[10] for s in data[sampleidx.+1]]
        Yω1 = [s[11] for s in data[sampleidx.+1]]
        Yω2 = [s[12] for s in data[sampleidx.+1]]
        Yω3 = [s[13] for s in data[sampleidx.+1]]

        pkernel = GaussianKernel(0.5, 1.5)
        qkernel = QuaternionKernel(0.5, ones(3))
        vkernel = GaussianKernel(0.5, 1.5)
        kernel = CompositeKernel([pkernel, qkernel, vkernel], [3, 4, 6])
        kernel = GeneralGaussianKernel(0.5, ones(13)*0.22)

        gprs = Vector{GaussianProcessRegressor}()
        for Y in [Yv1, Yv2, Yv3, Yω1, Yω2, Yω3]
            push!(gprs, GaussianProcessRegressor(X, Y, copy(kernel)))
        end
        mogpr = MOGaussianProcessRegressor(gprs)
        optimize!(mogpr, verbose=false)

        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
        states = getstates(mechanism)
        for i in 2:length(storage.x[1])
            μ = GPR.predict(mogpr, [SVector(reduce(vcat, states)...)])[1][1]
            v, ω = [SVector(μ[1:3]...)], [SVector(μ[4:6]...)]
            projectv!(v, ω, mechanism)
            foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
            states = getstates(mechanism)
            overwritestorage(storage, states, i)
        end

        onestep_mse = onesteperror(mechanism, storage)
        println("One step mean squared error: $onestep_mse")
        push!(onestep_msevec, onestep_mse)
    catch e
        display(e)
    end
end
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
println("Best one step mean squared error: $(minimum(onestep_msevec))")
plt = plot(1:length(onestep_msevec), onestep_msevec, seriestype = :scatter, yaxis=:log, title="Pendulum position one step MSE ", label = "pos MSE")
xlabel!("Trial")
savefig(plt, "2Dpendulum_GeneralGaussianKernel")