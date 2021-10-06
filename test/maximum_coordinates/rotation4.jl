using GPR
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))

storage, mech = simplependulum3D()

xfull = [q for q in storage.q[1]]
steps = 11
x = xfull[2:steps:end-1]
y = [q2/q1 for (q1, q2) in zip(xfull[2:steps:end-1], xfull[3:steps:end])]

gprvec = Vector{GaussianProcessRegressor}()
for i in 1:4
    N = length(x)
    xmat = Matrix{Float64}(undef, 4, N)
    ymat = Matrix{Float64}(undef, 1, N)
    for j in 1:N
        xmat[:,j] = quaternion_to_array(quaternion_projection(x[j]))
        ymat[1,j] = quaternion_to_array(quaternion_projection(y[j]))[i]
    end
    gpr = GaussianProcessRegressor(xmat,ymat,GaussianKernel(0.5,0.5), noisevariance = 0)
    optimize!(gpr)
    push!(gprvec, gpr)
end

function predict_quat(gprvec, quat)
    qarray = quaternion_to_array(quaternion_projection(quat))
    quatvec = zeros(4)
    for i in 1:4
        quatvec[i] = predict(gprvec[i], qarray)[1][1]
    end
    return quaternion_projection(UnitQuaternion(quatvec[1], quatvec[2:end]))
end

q = predict_quat(gprvec, x[1])
μ = q*x[1]
storage.q[1][2] = q*x[1]

for i in 2:999
    global q = predict_quat(gprvec, μ)
    global μ = q*μ
    storage.q[1][i+1] = μ
end
ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
