using Statistics
using GPR
using Plots

include("generatedata.jl")


storage = simple_pendulum()

function createdata(storage::Storage)
    # X = [x, y, vy, vy]'
    nelements = Int(length(storage.x[1])*0.02)
    X = Matrix{Float64}(undef, 4, nelements)
    for i in 1:nelements
        sidx = 1+(i-1)*10
        X[1,i] = storage.x[1][sidx][2]
        X[2,i] = storage.x[1][sidx][3]
        X[3,i] = storage.v[1][sidx][2]
        X[4,i] = storage.v[1][sidx][3]
    end
    return X
end

function createtargets(storage::Storage, idx::Int)
    # 1 = x[2], 2 = x[3], 3 = v[2], 4 = v[3]
    nelements = Int(length(storage.x[1])*0.02)
    @assert 1+nelements*10 <= length(storage.x[1])
    Y = Matrix{Float64}(undef, 1, nelements)
    rawdata = idx < 3 ? storage.x[1] : storage.v[1]
    for i in 1:nelements
        sidx = 1 + i*10  # shift targets by 10 timesteps
        Y[1, i] = rawdata[sidx][3-idx%2]
    end
    return Y
end

X = createdata(storage)

# compute, store and substract means from targets
Y = Matrix{Float64}(undef, 4, size(X,2))
Ymean = Vector{Float64}(undef, 4)
for i in 1:4
    Y[i,:] = createtargets(storage, i)
    Ymean[i] = mean(Y[i,:])
    Y[i,:] .-= Ymean[i]
end

println("Gridsearch for highest log marginal likelihood")
gprs = Vector{GaussianProcessRegressor}(undef, 4)
for i in 1:4
    bestlogPY = -Inf
    for σ in 0.01:0.05:2, l in 0.01:0.1:2, noise in 0:0.1:1
        kernel = GaussianKernel(σ, l)
        gpr = GaussianProcessRegressor(X, Y[i,:], kernel, noise)
        if gpr.logPY > bestlogPY
            gprs[i] = gpr
            bestlogPY = gpr.logPY
        end
    end
end

ntotal = length(storage.x[1])
Xtotal = Matrix{Float64}(undef, 4, ntotal)
for i in 1:ntotal
    Xtotal[1,i] = storage.x[1][i][2]
    Xtotal[2,i] = storage.x[1][i][3]
    Xtotal[3,i] = storage.v[1][i][2]
    Xtotal[4,i] = storage.v[1][i][3]
end

μx, σx = predict(gprs[1], Xtotal)
μx .+= Ymean[1]

plot_gp(Xtotal[1,:], μx, σx)
scatter!(reshape(X[1,:],:,1), reshape(Y[1,:] .+ Ymean[1],:,1), lab="Support points", legend = :topleft)

# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
