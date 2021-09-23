using Statistics
using GPR
using Plots
using LinearAlgebra

include("generatedata.jl")


storage, mech = simplependulum()

function create_maxc_data(storage::Storage, lag::Int = 1, step::Int = 50)
    # X = [x, y, vy, vy]'
    # lag: timesteps X should be predicted into the future
    # step: Sampling steps
    selements = length(storage.x[1])
    nelements = length(range(1, stop=selements-lag, step=step))
    X = Matrix{Float64}(undef, 4, nelements)
    Y = Matrix{Float64}(undef, 4, nelements)
    for i in 1:4
        rawdata = i < 3 ? storage.x[1] : storage.v[1]
        for (j,k) in enumerate(range(1,stop=selements-lag, step=step))
            X[i,j] = rawdata[k][3-i%2]
            Y[i,j] = rawdata[k+lag][3-i%2]
        end
    end
    return X, Y
end

X, Y = create_maxc_data(storage)

println("Gridsearch for highest log marginal likelihood")
kernels = Vector{GaussianKernel}(undef, 4)

for i in 1:4
    bestlogPY = -Inf
    for σ in 0.01:0.1:10, l in 0.01:0.1:10
        kernel = GaussianKernel(σ,l)
        trialgpr = GaussianProcessRegressor(X, Y[i,:], kernel)
        if trialgpr.logPY > bestlogPY
            kernels[i] = kernel
            bestlogPY = trialgpr.logPY
        end
    end
    println("Kernel: $i σ: $(kernels[i].σ) l: $(kernels[i].l)")
end
mo_gpr = MOGaussianProcessRegressor(X, Y, kernels)

println("Gridsearch done, predicting...")
xstart = SVector{size(X,1),Float64}(X[:,1])
μstatic, σstatic = predict(mo_gpr, xstart, 200)
Xtotal, _ = create_maxc_data(storage, 1, 1)

μ = zeros(length(μstatic[1]), length(μstatic))
σ = zeros(length(σstatic[1]), length(σstatic))
for i in 1:length(μstatic)
    μ[:,i] = μstatic[i]
    σ[:,i] = σstatic[i]
end

xtrue = Xtotal[:,2:end]
display(μ)

error = μ[1:2,1:200] .- xtrue[1:2,1:200]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

plot_gp(μ[1,:], μ[2,:], sqrt.(σ[1,:].^2 + σ[2,:].^2))
scatter!(reshape(X[1,:],:,1), reshape(X[2,:],:,1), lab="Support points")

# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
