using GPR
using ConstrainedDynamicsVis
using Plots
using StatsBase: sample

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


storage, mechanism, initialstates = simplependulum2D()
gtdata = loaddata(storage)
data = deepcopy(gtdata)
cleardata!(data)

nsamples = Int(round(length(data)/5))
nruns = 30

mesvec = Vector{Float64}()
for _ in 1:nruns
    try
        resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position
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

        state = getstate(mechanism)
        project = true
        for idx in 2:length(storage.x[1])
            global state
            μ = GPR.predict(mogpr, [state])[1][1]
            v, ω = SVector(μ[1:3]...), SVector(μ[4:6]...)
            projectv!([v], [ω], mechanism)
            state = getstate(mechanism)
            overwritestorage(storage, state, idx)
        end

        mse = 0
        for i in 1:length(storage.x[1])
            mse += sum(sum((gtdata[i][1:3]-storage.x[1][i]).^2))/3length(data)
        end
        println("Mean squared error: $mse")
        push!(mesvec, mse)
    catch
    end
end
# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")

plt = plot(1:length(mesvec), mesvec, seriestype = :scatter, yaxis=:log, title="Pendulum position MSE ", label = "pos MSE")
xlabel!("Trial")
savefig(plt, "2Dpendulum_GeneralGaussianKernel")