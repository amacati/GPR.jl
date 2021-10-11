using GPR
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))


storage, mech = simplependulum3D()
steps = 11
Δt = 0.01
v = [vi for vi in storage.v[1][1:steps:end-1]]
vfuturex = [vj[1] for vj in storage.v[1][2:steps:end]]
vfuturey = [vj[2] for vj in storage.v[1][2:steps:end]]
vfuturez = [vj[3] for vj in storage.v[1][2:steps:end]]
ω = [ωi for ωi in storage.ω[1][1:steps:end-1]]
ωfuturex = [ωj[1] for ωj in storage.ω[1][2:steps:end]]
ωfuturey = [ωj[2] for ωj in storage.ω[1][2:steps:end]]
ωfuturez = [ωj[3] for ωj in storage.ω[1][2:steps:end]]

kernel = GaussianKernel(0.5,1.5)
gpvx = optimize!(GaussianProcessRegressor(v, vfuturex, copy(kernel)))
gpvy = optimize!(GaussianProcessRegressor(v, vfuturey, copy(kernel)))
gpvz = optimize!(GaussianProcessRegressor(v, vfuturez, copy(kernel)))
gpωx = optimize!(GaussianProcessRegressor(ω, ωfuturex, copy(kernel)))
gpωy = optimize!(GaussianProcessRegressor(ω, ωfuturey, copy(kernel)))
gpωz = optimize!(GaussianProcessRegressor(ω, ωfuturez, copy(kernel)))

μvx = predict(gpvx, v[1])[1][1]
μvy = predict(gpvy, v[1])[1][1]
μvz = predict(gpvz, v[1])[1][1]
μωx = predict(gpωx, ω[1])[1][1]
μωy = predict(gpωy, ω[1])[1][1]
μωz = predict(gpωz, ω[1])[1][1]
μv = SVector(μvx, μvy, μvz)
μω = SVector(μωx, μωy, μωz)

storage.v[1][2] = μv
storage.x[1][2] = updatestate(storage.x[1][1], μv, Δt)
storage.ω[1][2] = μω
storage.q[1][2] = updatestate(storage.q[1][1], μω, Δt)

for i in 2:999
    global μvx = predict(gpvx, μv)[1][1]
    global μvy = predict(gpvy, μv)[1][1]
    global μvz = predict(gpvz, μv)[1][1]
    global μωx = predict(gpωz, μω)[1][1]
    global μωx = predict(gpωx, μω)[1][1]
    global μωx = predict(gpωy, μω)[1][1]
    global μv = SVector(μvx, μvy, μvz)
    global μω = SVector(μωx, μωy, μωz)

    storage.v[1][2] = μv
    storage.x[1][i+1] = updatestate(storage.x[1][i], μv, Δt)
    storage.ω[1][2] = μω
    storage.q[1][i+1] = updatestate(storage.q[1][i], μω, Δt)
end
ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
