using Statistics
using ConstrainedDynamics
using ConstrainedDynamicsVis
using GPR
using Plots

joint_axis = [1.0; 0.0; 0.0]

Δt=0.01
g = -9.81
m = 1.0
l = 1.0
r = 0.01

p2 = [0.0;0.0;l / 2] # joint connection point

# Links
origin = Origin{Float64}()
link1 = Cylinder(r, l, m)

# Constraints
joint_between_origin_and_link1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=p2))

links = [link1]
constraints = [joint_between_origin_and_link1]


mech = Mechanism(origin, links, constraints, g=g,Δt=Δt)

T0 = 2*pi*sqrt((m*l^2/3)/(-g*l/2))
t0 = 1

q1 = UnitQuaternion(RotX(π / 2))
setPosition!(origin,link1,p2 = p2,Δq = q1)
setVelocity!(link1)
storage = simulate!(mech,10.,record = true)










function createdata(storage::Storage)::Matrix{Float64}
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

function createtargets(storage::Storage, idx::Int)::Matrix{Float64}
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
Yx = createtargets(storage, 1)
Yy = createtargets(storage, 2)
Yvx = createtargets(storage, 3)
Yvy = createtargets(storage, 4)

# compute, store and substract means from targets
Yx_mean = mean(Yx)
Yy_mean = mean(Yy)
Yvx_mean = mean(Yvx)
Yvy_mean = mean(Yvy)

Yx .-= Yx_mean
Yy .-= Yy_mean
Yvx .-= Yvx_mean
Yvy .-= Yvy_mean

# create GPR, predict values for Y
kernel_x = GaussianKernel(0.2,0.7)

ntotal = length(storage.x[1])
Xtotal = Matrix{Float64}(undef, 4, ntotal)
for i in 1:ntotal
    Xtotal[1,i] = storage.x[1][i][2]
    Xtotal[2,i] = storage.x[1][i][3]
    Xtotal[3,i] = storage.v[1][i][2]
    Xtotal[4,i] = storage.v[1][i][3]
end
gpr_x = GaussianProcessRegressor(X, Yx, kernel_x)
μx, σx = predict(gpr_x, Xtotal)
μx .+= Yx_mean

# mse for Ys

# plot data


plot_gp(Xtotal[1,:], μx, σx)
scatter!(reshape(X[1,:],:,1), reshape(Yx .+ Yx_mean,:,1), lab="Support points", legend = :topleft)











# ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
