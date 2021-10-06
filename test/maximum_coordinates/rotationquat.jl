using GPR
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))

storage, mech = simplependulum3D()

xfull = [q for q in storage.q[1]]
steps = 11
x = xfull[2:steps:end-1]
y = [q2/q1 for (q1, q2) in zip(xfull[2:steps:end-1], xfull[3:steps:end])]

gpqr = GaussianProcessQuaternionRegressor(x, y, QuaternionKernel(0.1,0.08), noisevariance = 0)
q = predict(gpqr, x[1])
display(gpqr.Ymean)
μ = q*x[1]
storage.q[1][2] = q*x[1]

for i in 2:999
    global q = predict(gpqr, μ)
    global μ = q*μ
    storage.q[1][i+1] = μ
end
ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
