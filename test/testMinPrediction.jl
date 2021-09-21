using GPR
using Plots
using Rotations
using Quaternions

include("generatedata.jl")


storage = simple_pendulum()


function qangle(q)
    return rotation_angle(q) * sign(q[3,2])
end

function create_minc_data(storage::Storage, lag::Int = 1, step::Int = 50)
    selements = length(storage.x[1])
    nelements = length(range(1, stop=selements-lag, step=step))
    X = Matrix{Float64}(undef, 2, nelements)
    Y = Matrix{Float64}(undef, 2, nelements)
    for (j,k) in enumerate(range(1,stop=selements-lag, step=step))
        X[1,j] = qangle(storage.q[1][k])
        Y[1,j] = qangle(storage.q[1][k+lag])
    end
    for (j,k) in enumerate(range(1,stop=selements-lag, step=step))
        X[2,j] = storage.ω[1][k][1]
        Y[2,j] = storage.ω[1][k+lag][1]
    end
    return X, Y

end

X, Y = create_minc_data(storage)

# compute, store and substract means from targets
Ymean = Vector{Float64}(undef, 2)
for i in 1:2
    Ymean[i] = mean(Y[i,:])
    Y[i,:] .-= Ymean[i]
end

println("Gridsearch for highest log marginal likelihood")
gprs = Vector{GaussianProcessRegressor}(undef, 2)

for i in 1:2
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
xtotal, _ = create_minc_data(storage, 1, 1)

function convert_to_cartesian(x)
    xcart = Matrix{Float64}(undef,2,size(x,2))
    for i in 1:size(x,2)
        xcart[:,i] = 0.5 .* [sin(x[1,i]), -cos(x[1,i])]
    end
    return xcart
end

xtrue = convert_to_cartesian(xtotal)[:,2:end]
xpredict = μ .+ Ymean
xpredict = convert_to_cartesian(xpredict)
error = xpredict[1:2,1:200] .- xtrue[1:2,1:200]
mse = sum(error.^2) / length(error)
println("Mean squared error: $mse")

plot_gp(μ[1,:] .+ Ymean[1], μ[2,:] .+ Ymean[2], sqrt.(σ[1,:].^2 + σ[2,:].^2))
# plot_gp(μx .+ Ymean[1], μy .+ Ymean[2], sqrt.(σx.^2 + σy.^2))
# plot!(x, μ, lw = 2, legend = :topleft, lab = "Simulated trajectory")

scatter!(reshape(X[1,:],:,1), reshape(X[2,:],:,1), lab="Support points")

# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
