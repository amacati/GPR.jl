using BenchmarkTools
using LinearAlgebra
using GPR
using StaticArrays

include(joinpath("..", "examples", "generatedata.jl"))
include(joinpath("..", "examples", "utils.jl"))


println("Gaussian Process projection benchmark.")
suite = BenchmarkGroup()

# Initialize sim, GPs, optimize kernels etc.
storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage)
cleardata!(data)
sampleidx = 1:5:length(data)
X = data[sampleidx]
Yv1 = [s[8] for s in data[sampleidx.+1]]
Yv2 = [s[9] for s in data[sampleidx.+1]]
Yv3 = [s[10] for s in data[sampleidx.+1]]
Yω1 = [s[11] for s in data[sampleidx.+1]]
Yω2 = [s[12] for s in data[sampleidx.+1]]
Yω3 = [s[13] for s in data[sampleidx.+1]]
kernel = GeneralGaussianKernel(0.5, ones(13)*0.22)
gprs = Vector{GaussianProcessRegressor}()
for Y in [Yv1, Yv2, Yv3, Yω1, Yω2, Yω3]
    push!(gprs, GaussianProcessRegressor(X, Y, copy(kernel)))
end
mogpr = MOGaussianProcessRegressor(gprs)
# optimize!(mogpr, verbose=false)

function projectionbenchmark(mechanism, initialstates, mogpr)
    resetMechanism!(mechanism, initialstates)  # Reset mechanism to starting position
    foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
    states = getstates(mechanism)
    for _ in 2:1000
        μ = GPR.predict(mogpr, [SVector(reduce(vcat, states)...)])[1][1]
        v, ω = [SVector(μ[1:3]...)], [SVector(μ[4:6]...)]
        projectv!(v, ω, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
        states = getstates(mechanism)
    end
end

projectionbenchmark(mechanism, initialstates, mogpr)

println("Setup complete, starting benchmark.")
suite[1] = @benchmark projectionbenchmark($mechanism, $initialstates, $mogpr)
display(suite[1])