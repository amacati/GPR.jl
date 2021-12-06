using GPR
using ConstrainedDynamics
using ConstrainedDynamicsVis
using Plots
using Random
using GaussianProcesses
using LineSearches
using Optim

include("utils/utils.jl")
include("utils/data/simulations.jl")
include("utils/data/datasets.jl")

g = 9.81
m = 1.
l = 1.
Is = 1/12*m*l^2 + m*(.5l)^2

function energy(θ, ω) where T
    Ekin = .5Is*ω^2
    Epot = m*g*.5l*(1-cos(θ))
    return Epot + Ekin
end

function explicitEuler(steps, Δt, θ, ω)
    dω(θ, ω) = return -1.5sin(θ)*g/l  # 0.5lmgsin(θ)/I
    dθ(θ, ω) = return ω
    θs, ωs = zeros(steps), zeros(steps)
    for i in 1:steps
        Δθ = dθ(θ, ω)*Δt
        Δω = dω(θ, ω)*Δt
        θ += Δθ
        ω += Δω
        θs[i], ωs[i] = θ, ω
    end
    return θs, ωs
end

function gpVariationalIntegration(steps, θ, nsamples)
    Δtsim = 0.0001
    offset = Int(0.01/Δtsim)
    traindf = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}())
    s = simplependulum2D(200*offset, Δt=Δtsim, θstart=θ)[1]
    for _ in 1:nsamples
        i = rand(1:length(s.x[1])-offset)
        push!(traindf, [getStates(s, i), getStates(s, i+offset)])
    end

    mechanism = simplependulum2D(1, Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    # Create train and testsets
    xtrain_old = reduce(hcat, [CState(x) for x in traindf.sold])

    xtrain_curr = [CState(x) for x in traindf.scurr]
    vωindices = [9, 10, 11]
    ytrain = [[cs[i] for cs in xtrain_curr] for i in vωindices]

    predictedstates = Vector{CState{Float64,1}}()
    params = Vector{Float64}()  # declare in outer scope
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)["P1_MAX16"]
    end
    gps = Vector{GPE}()
    cache = MDCache()
    for (id, yi) in enumerate(ytrain)
        kernel = SEArd(log.(params[2:end]), log(params[1]))
        mean = MeanZero()
        gp = GP(xtrain_old, yi, mean, kernel)
        GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
        push!(gps, gp)
    end
    
    oldstates = CState([0, .5sin(θ)l, -.5cos(θ)l, q2vec(UnitQuaternion(RotX(θ)))..., zeros(6)...])
    setstates!(mechanism, oldstates)
    getvω(μ) = return [SVector(0,μ[1:2]...)], [SVector(μ[3],0,0)]
    for _ in 1:steps
        obs = reshape(oldstates.state, :, 1)
        μ = [predict_y(gp, obs)[1][1] for gp in gps]
        vcurr, ωcurr = getvω(μ)
        projectv!(vcurr, ωcurr, mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        oldstates = CState(mechanism)
        push!(predictedstates, oldstates)
    end
    θs, ωs = zeros(steps), zeros(steps)
    for (i, cstate) in enumerate(predictedstates)
        q = UnitQuaternion(cstate[4], cstate[5:7])
        θs[i] = Rotations.rotation_angle(q)*sign(q.x)*sign(q.w)  # Signum for axis direction
        ωs[i] = cstate[11]
    end
    return θs, ωs
end

function visualizePendulum(θs)
    mechanism = simplependulum2D(1)[2]
    s = Storage{Float64}(length(θs), 1)
    for i in 1:length(θs)
        x = [0, .5sin(θs[i]), -.5cos(θs[i])]
        q = UnitQuaternion(RotX(θs[i]))
        overwritestorage(s, CState([x..., q2vec(q)..., zeros(6)...]), i)
    end
    ConstrainedDynamicsVis.visualize(mechanism, s; showframes = true, env = "editor")
end

trueenergy = m*g*.5l*(1-cos(π/2))

θeuler, ωeuler = explicitEuler(100*60*100, 0.01, π/2, 0)
θgp, ωgp = gpVariationalIntegration(100*60*100, π/2, 20)
plt = plot([1:length(θeuler)]./100, [[abs(energy(θ, ω)/trueenergy-1)*100 for (θ, ω) in zip(θeuler, ωeuler)], [abs(energy(θ, ω)/trueenergy-1)*100 for (θ, ω) in zip(θgp, ωgp)]],
           title="Energy change comparison", label = ["Explicit Euler" "Implicit GP 20 samples"], xlabel="Time in seconds", ylabel = "Relative % error of total energy")
savefig(plt, "energyComparison100m")
           
θeuler, ωeuler = explicitEuler(100*10, 0.01, π/2, 0)
θgp, ωgp = gpVariationalIntegration(100*10, π/2, 20)
θgp2, ωgp2 = gpVariationalIntegration(100*10, π/2, 10)
plt = plot([1:length(θeuler)]./100, [[abs(energy(θ, ω)/trueenergy-1)*100 for (θ, ω) in zip(θeuler, ωeuler)], [abs(energy(θ, ω)/trueenergy-1)*100 for (θ, ω) in zip(θgp, ωgp)], [abs(energy(θ, ω)/trueenergy-1)*100 for (θ, ω) in zip(θgp2, ωgp2)]],
            title="Energy change comparison", label = ["Explicit Euler" "Implicit GP 20 samples" "Implicit GP 10 samples"], xlabel="Time in seconds", ylabel = "Relative % error of total energy")
savefig(plt, "energyComparison10s")
