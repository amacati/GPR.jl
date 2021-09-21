using Statistics
using GPR
using Plots
using Rotations

include("generatedata.jl")


storage = simple_pendulum()

function create_maxc_data(storage::Storage, lag::Int = 1, step::Int = 5)
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

function create_minc_data(storage::Storage, lag::Int = 1, step::Int = 5)
    selements = length(storage.x[1])
    nelements = length(range(1, stop=selements-lag, step=step))
    X = Matrix{Float64}(undef, 2, nelements)
    Y = Matrix{Float64}(undef, 2, nelements)
    for (j,k) in enumerate(range(1,stop=selements-lag, step=step))
        X[1,j] = rotation_angle(storage.q[1][k])
        Y[1,j] = rotation_angle(storage.q[1][k+lag])
    end
    for (j,k) in enumerate(range(1,stop=selements-lag, step=step))
        X[2,j] = storage.ω[1][k][1]
        Y[2,j] = storage.ω[1][k+lag][1]
    end
    return X, Y

end

X, Y = create_minc_data(storage)
display(X)
display(Y)
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
    for σ in 0.01:0.1:2, l in 0.01:0.2:5
        kernel = MaternKernel{1.5}(σ,l)
        gpr = GaussianProcessRegressor(X, Y[i,:], kernel)
        if gpr.logPY > bestlogPY
            gprs[i] = gpr
            bestlogPY = gpr.logPY
        end
    end
    # println("Kernel: $i σ: $(gprs[i].kernel.σ) l: $(gprs[i].kernel.l)")
    println("Kernel: $i σ: $(gprs[i].kernel.v) l: $(gprs[i].kernel.l)")
end
println("Gridsearch done, predicting...")

#=
for i in 1:4
    kernel = MaternKernel{1.5}(0.5,3)
    gprs[i] = GaussianProcessRegressor(X,Y[i, :], kernel)
end
=#
ntotal = length(storage.x[1])
xstart = X[:,1]
μ, σ = predict(gprs, xstart, 200)
Xtotal, _ = create_maxc_data(storage, 1, 1)
μx, σx = predict(gprs[1], Xtotal[:,1:10])
μy, σy = predict(gprs[2], Xtotal[:,1:10])


plot_gp(μ[1,:] .+ Ymean[1], μ[2,:] .+ Ymean[2], sqrt.(σ[1,:].^2 + σ[2,:].^2))
# plot_gp(μx .+ Ymean[1], μy .+ Ymean[2], sqrt.(σx.^2 + σy.^2))
# plot!(x, μ, lw = 2, legend = :topleft, lab = "Simulated trajectory")
scatter!(reshape(X[1,:],:,1), reshape(X[2,:],:,1), lab="Support points")

# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
