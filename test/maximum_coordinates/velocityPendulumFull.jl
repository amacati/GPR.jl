using GPR
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))


storage, mech = simplependulum3D()
steps = 11
x = [xi for xi in storage.x[1][1:steps:end-1]]
q = [qi for qi in storage.q[1][1:steps:end-1]]
v = [vi for vi in storage.v[1][1:steps:end-1]]
ω = [ωi for ωi in storage.ω[1][1:steps:end-1]]
states = [vcat(xi, qi, vi, ωi) for (xi, qi, vi, ωi) in zip(x, q, v, ω)]

vfuturex = [vj[1] for vj in v[2:steps:end]]
vfuturey = [vj[2] for vj in v[2:steps:end]]
vfuturez = [vj[3] for vj in v[2:steps:end]]
ωfuturex = [ωj[1] for ωj in storage.ω[1][2:steps:end]]
ωfuturey = [ωj[2] for ωj in storage.ω[1][2:steps:end]]
ωfuturez = [ωj[3] for ωj in storage.ω[1][2:steps:end]]

kernel = GaussianKernel(0.5,1.5)
gpvx = GaussianProcessRegressor(states, vfuturex, kernel)
gpvx = GaussianProcessRegressor(states, vfuturey, kernel)
gpvx = GaussianProcessRegressor(states, vfuturez, kernel)
gpωx = GaussianProcessRegressor(states, ωfuturex, kernel)
gpωx = GaussianProcessRegressor(states, ωfuturey, kernel)
gpωx = GaussianProcessRegressor(states, ωfuturez, kernel)

