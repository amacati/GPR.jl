using GPR
using ConstrainedDynamics: foreachactive, updatestate!
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


storage, mechanism, initialstates = doublependulum3D()
data = loaddata(storage)
resetMechanism!(mechanism, initialstates)  # Reset mechanism to starting position

steps = 11
start = 1

X = data[start:steps:end-1]
Yv11 = [s[8] for s in data[start+1:steps:end]]
Yv12 = [s[9] for s in data[start+1:steps:end]]
Yv13 = [s[10] for s in data[start+1:steps:end]]
Yv21 = [s[21] for s in data[start+1:steps:end]]
Yv22 = [s[22] for s in data[start+1:steps:end]]
Yv23 = [s[23] for s in data[start+1:steps:end]]
Yω11 = [s[11] for s in data[start+1:steps:end]]
Yω12 = [s[12] for s in data[start+1:steps:end]]
Yω13 = [s[13] for s in data[start+1:steps:end]]
Yω21 = [s[24] for s in data[start+1:steps:end]]
Yω22 = [s[25] for s in data[start+1:steps:end]]
Yω23 = [s[26] for s in data[start+1:steps:end]]

kernel = GeneralGaussianKernel(500., ones(26)*0.001)
gprs = Vector{GaussianProcessRegressor}()
for Y in [Yv11, Yv12, Yv13, Yv21, Yv22, Yv23, Yω11, Yω12, Yω13, Yω21, Yω22, Yω23]
    push!(gprs, GaussianProcessRegressor(X, Y, copy(kernel)))
end
mogpr = MOGaussianProcessRegressor(gprs)
# optimize!(mogpr, verbose=false)


foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # New position, old velocity
states = getstates(mechanism)
for i in 2:1000
    μ = GPR.predict(mogpr, [SVector(reduce(vcat, states)...)])[1][1]
    v, ω = [SVector(μ[1:3]...), SVector(μ[4:6]...)], [SVector(μ[7:9]...), SVector(μ[10:12]...)]
    projectv!(v, ω, mechanism)
    foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
    states = getstates(mechanism)
    overwritestorage(storage, states, i)
end

mse = onesteperror(mechanism, storage)
# resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position

println("Mean squared error: $mse")

# ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
