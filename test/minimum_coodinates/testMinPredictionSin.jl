using GPR
using Plots
using Rotations
using Quaternions
using Statistics

include(joinpath("..", "generatedata.jl"))


storage, _ = simplependulum()

function qangle(q)
    return rotation_angle(q) * sign(q[3,2])
end

function create_minc_data(storage::Storage, lag::Int = 1, step::Int = 50)
    selements = length(storage.x[1])
    nelements = length(range(1, stop=selements-lag, step=step))
    X = Matrix{Float64}(undef, 3, nelements)
    Y = Matrix{Float64}(undef, 3, nelements)
    for (j,k) in enumerate(range(1,stop=selements-lag, step=step))
        X[1,j] = qangle(storage.q[1][k])
        Y[1,j] = qangle(storage.q[1][k+lag])
        X[2,j] = storage.ω[1][k][1]
        Y[2,j] = storage.ω[1][k+lag][1]
        X[3,j] = sin(X[1,j])
        Y[3,j] = sin(Y[1,j])
    end
    return X, Y
end

X, Y = create_minc_data(storage)

kernels = Vector{GeneralGaussianKernel}(undef, 3)

for i in 1:3
    kernels[i] = GeneralGaussianKernel(0.5,ones(3)*0.5)
end

mo_gpr = MOGaussianProcessRegressor(X, Y, kernels)
optimize!(mo_gpr)

xstart = X[:,1]
xstart = SVector{size(X,1),Float64}(X[:,1])
μstatic, σstatic = predict(mo_gpr, xstart, 200)
xtotal, _ = create_minc_data(storage, 1, 1)

function convert_to_cartesian(x)
    xcart = Matrix{Float64}(undef,2,size(x,2))
    for i in 1:size(x,2)
        xcart[:,i] = 0.5 .* [sin(x[1,i]), -cos(x[1,i])]
    end
    return xcart
end

xtrue = convert_to_cartesian(xtotal)[:,2:end]
μ = zeros(length(μstatic[1]), length(μstatic))
σ = zeros(length(μstatic[1]), length(μstatic))
for i in 1:length(μstatic)
    μ[:,i] = μstatic[i]
    σ[:,i] = σstatic[i]
end
μcart = convert_to_cartesian(μ)

error = μcart[1:2,1:200] .- xtrue[1:2,1:200]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

plot_gp(μ[1,:], μ[2,:], sqrt.(σ[1,:].^2 + σ[2,:].^2))
scatter!(reshape(X[1,:],:,1), reshape(X[2,:],:,1), lab="Support points")

# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
