using GPR
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))

storage, mech = simplependulum3D()

x = [xi for xi in storage.x[1]]
q = [qi for qi in storage.q[1]]
allstates = [vcat(xi,quaternion_to_array(quaternion_projection(qi))) for (xi,qi) in zip(x,q)]
steps = 11
states = allstates[2:steps:end-1]
yx = [xj[1] - xi[1] for (xi, xj) in zip(x[2:steps:end-1], x[3:steps:end])]
yy = [xj[2] - xi[2] for (xi, xj) in zip(x[2:steps:end-1], x[3:steps:end])]
yz = [xj[3] - xi[3] for (xi, xj) in zip(x[2:steps:end-1], x[3:steps:end])]
yq = [q2/q1 for (q1, q2) in zip(q[2:steps:end-1], q[3:steps:end])]

poskernel = GaussianKernel(0.5,1.5)
qkernel = QuaternionKernel(0.1,0.08)
ckernel = CompositeKernel([poskernel, qkernel], [3, 4])
gprx = GaussianProcessRegressor(states, yx, ckernel)
gpry = GaussianProcessRegressor(states, yy, ckernel)
gprz = GaussianProcessRegressor(states, yz, ckernel)
gpqr = GaussianProcessQuaternionRegressor(states, yq, ckernel, noisevariance = 0)

x = predict(gprx, states[1])[1][1]
y = predict(gpry, states[1])[1][1]
z = predict(gprz, states[1])[1][1]
q = predict(gpqr, states[1])

μpos = storage.x[1][1] + SVector(x, y, z)
μq = q*storage.q[1][1]

storage.x[1][2] = μpos
storage.q[1][2] = μq
μ = vcat(μpos, quaternion_to_array(quaternion_projection(μq)))

for i in 2:999
    global x = predict(gprx, μ)[1][1]
    global y = predict(gpry, μ)[1][1]
    global z = predict(gprz, μ)[1][1]
    global q = predict(gpqr, μ)

    global μpos = μpos + SVector(x, y, z)
    global μq = q*μq
    storage.x[1][i+1] = μpos
    storage.q[1][i+1] = μq
    global μ = vcat(SVector(x, y, z), quaternion_to_array(quaternion_projection(μq)))
end
ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
