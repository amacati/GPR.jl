using Statistics
using GPR
using Plots
using LinearAlgebra

include("generatedata.jl")


storage = simple_pendulum()

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

# compute, store and substract means from targets
Ymean = Vector{Float64}(undef, 4)
for i in 1:4
    Ymean[i] = mean(Y[i,:])
    Y[i,:] .-= Ymean[i]
end

println("Gridsearch for highest log marginal likelihood")
gprs = Vector{GaussianProcessRegressor}(undef, 4)

for i in 1:4
    bestlogPY = -Inf
    for σ in 0.01:0.1:10, l in 0.01:0.1:10
        kernel = GaussianKernel(σ,l)
        trialgpr = GaussianProcessRegressor(X, Y[i,:], kernel)
        if trialgpr.logPY > bestlogPY
            gprs[i] = trialgpr
            bestlogPY = trialgpr.logPY
        end
    end
    println("Kernel: $i σ: $(gprs[i].kernel.σ) l: $(gprs[i].kernel.l)")
    # println("Kernel: $i σ: $(gprs[i].kernel.v) l: $(gprs[i].kernel.l)")
end
println("Gridsearch done, predicting...")

xstart = X[:,1]
μ, σ = predict(gprs, xstart, 200, Ymean)
Xtotal, _ = create_maxc_data(storage, 1, 1)

xtrue = Xtotal[:,2:end]
display(xtrue[1:2,:])
xpredict = μ .+ Ymean
error = μ[1:2,1:200] .- xtrue[1:2,1:200]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

plot_gp(xpredict[1,:], xpredict[2,:], sqrt.(σ[1,:].^2 + σ[2,:].^2))

#=
μx, σx = predict(gprs[1], Xtotal[:,1:200])
μy, σy = predict(gprs[2], Xtotal[:,1:200])
plot_gp(μx .+ Ymean[1], μy .+ Ymean[2], sqrt.(σx.^2 + σy.^2))
=#

# plot!(x, μ, lw = 2, legend = :topleft, lab = "Simulated trajectory")
scatter!(reshape(X[1,:],:,1), reshape(X[2,:],:,1), lab="Support points")

# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
