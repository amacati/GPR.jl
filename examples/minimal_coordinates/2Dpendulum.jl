using GPR
using ConstrainedDynamicsVis

include(joinpath("..", "generatedata.jl"))
include(joinpath("..", "utils.jl"))


storage, mechanism, initialstates = simplependulum2D()
data = loaddata(storage)
resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position
steps = 30

function vec2state(x)
    state = Vector()
    for i in 0:13:length(x)-1
        s = ConstrainedDynamics.State{Float64}()
        s.xc = x[1+i:3+i]
        s.qc = UnitQuaternion(x[4+i], x[5+i:7+i])
        s.vc = x[8+i:10+i]
        s.ωc = x[11+i:13+i]
        push!(state, s)
    end
    return state
end

function max2mincoordinates(data, mechanism)
    mindata = Vector{SVector}()
    for state in data
        state = vec2state(state)
        resetMechanism!(mechanism, state)
        minstate = Vector{Float64}()
        for eqc in mechanism.eqconstraints
            append!(minstate, ConstrainedDynamics.minimalCoordinates(mechanism, eqc))
            append!(minstate, ConstrainedDynamics.minimalVelocities(mechanism, eqc))
        end
        push!(mindata, SVector(minstate...))
    end
    return mindata
end

function min2maxcoordinates(data, mechanism)
    maxdata = Vector{Vector{Float64}}()
    for state in data
        maxstate = Vector{Float64}()
        for eqc in mechanism.eqconstraints
            ConstrainedDynamics.setPosition!(mechanism, eqc, [state[1]])
            ConstrainedDynamics.setVelocity!(mechanism, eqc, [state[2]])
            append!(maxstate, getstate(mechanism))
        end
        push!(maxdata, maxstate)
    end
    return maxdata
end

mindata = max2mincoordinates(data, mechanism)
resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position

X = mindata[1:steps:end-1]
Yθ = [sample[1] for sample in mindata[2:steps:end]]
Yω = [sample[2] for sample in mindata[2:steps:end]]

kernel = GeneralGaussianKernel(0.5, ones(2)*0.2)
gprθ = GaussianProcessRegressor(X, Yθ, copy(kernel))
gprω = GaussianProcessRegressor(X, Yω, copy(kernel))
gprs = [gprθ, gprω]
Threads.@threads for gpr in gprs
    optimize!(gpr)
end

function predictθω(gprs, state)
    θ = predict(gprs[1], state)[1][1]
    ω = predict(gprs[2], state)[1][1]
    return θ, ω
end

state = getstate(mechanism)
state = max2mincoordinates([state], mechanism)[1]
for idx in 2:1000
    global state
    θ, ω = predictθω(gprs, state)
    state = SVector(θ, ω)
    maxstate = min2maxcoordinates([state], mechanism)[1]
    overwritestorage(storage, maxstate, idx)
end

mse = 0
for i in 1:length(data)
    mse += sum(sum((data[i][1:3]-storage.x[1][i]).^2))/3length(data)
end
println("Mean squared error: $mse")

ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")