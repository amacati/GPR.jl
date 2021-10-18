using GPR
using Plots
using Rotations
using Quaternions
using Statistics

include(joinpath("..", "generatedata.jl"))


storage, mech = simplependulum2D()
data = load2Ddata(storage, coord="min")
X = data[1][:,1:50:end-1]
X = vcat(X, reshape(sin.(X[1,:]),1,:))  # Add sin(θ) which acts as quaternion in 2D
Y = data[1][:,2:50:end]
Y = vcat(Y, reshape(sin.(Y[1,:]),1,:))

kernels = [GeneralGaussianKernel(0.5,ones(size(X,1))*0.5) for _ in 1:size(X,1)]
mo_gpr = MOGaussianProcessRegressor(X, Y, kernels)
optimize!(mo_gpr)

xstart = SVector{size(X,1),Float64}(X[:,1])
μ, σ = predict(mo_gpr, xstart, 999)

function convert_to_cartesian(x)
    xcart = Matrix{Float64}(undef,2,size(x,2))
    for i in 1:size(x,2)
        xcart[:,i] = 0.5 .* [sin(x[1,i]), -cos(x[1,i])]
    end
    return xcart
end

xtrue = convert_to_cartesian(data[1][:,2:end])
μcart = convert_to_cartesian(reshape(reinterpret(Float64, μ), (3,:)))

error = μcart[1:2,1:999] .- xtrue[1:2,1:999]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

function transformTo3D(X)
    ret = [[zeros(length(X[1][1])) for _ in 1:length(X[1])] for _ in 1:length(X)]
    for id in 1:length(X)
        for t in 1:length(X[1])
            state = X[id][t]
            q = UnitQuaternion(RotX(state[1]))
            ret[id][t] = [0, 0.5*sin(state[1]), -0.5*cos(state[1]), q.w, q.x, q.y, q.z]
        end
    end
    return ret
end

visualize_prediction(mech, transformTo3D([μ]))
