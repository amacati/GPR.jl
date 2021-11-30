using BenchmarkTools
using LinearAlgebra
using GPR
using StaticArrays
using Random
using LineSearches
using Optim

include("../examples/utils.jl")
include("../examples/generatedata.jl")
include("../examples/generate_datasets.jl")
include("../examples/mDynamics.jl")
include("../examples/predictdynamics.jl")

println("Gaussian Process inference benchmark.")
suite = BenchmarkGroup()

traindfs, testdfs = loaddatasets("FBfriction")  # Each thread operates on its own dataset -> no races
traindf = traindfs.df[1][shuffle(1:nrow(traindfs.df[1]))[1:100], :]
testdf = testdfs.df[1][shuffle(1:nrow(testdfs.df[1]))[1:1], :]
mechanism = fourbar(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism. Assume perfect knowledge of J and M

xtest_future_true = [CState(x) for x in testdf.sfuture]
# Create train and testsets
xtrain_old = reduce(hcat, [CState(x) for x in traindf.sold])
xtrain_curr = [CState(x) for x in traindf.scurr]
vωindices = [9, 10, 22, 23, 35, 36, 48, 49, 11, 24, 37, 50]  # v12, v13, v22, v23, v32, v33, v42, v43, ω11, ω21, ω31, ω41
ytrain = [[s[i] for s in xtrain_curr] for i in vωindices]
xtest_old = [CState(x) for x in testdf.sold]

predictedstates = Vector{CState{Float64,4}}()
params = Vector{Float64}()  # declare in outer scope
open(joinpath(dirname(dirname(@__FILE__)), "examples", "config", "config.json"),"r") do f
    global params = JSON.parse(f)["FB_MAX128"]
end
gps = Vector{GPE}()
function _getμfb(mech)
    v12 = mechanism.bodies[1].state.vsol[2][2]
    v13 = mechanism.bodies[1].state.vsol[2][3]
    v22 = mechanism.bodies[2].state.vsol[2][2]
    v23 = mechanism.bodies[2].state.vsol[2][3]
    v32 = mechanism.bodies[3].state.vsol[2][2]
    v33 = mechanism.bodies[3].state.vsol[2][3]
    v42 = mechanism.bodies[4].state.vsol[2][2]
    v43 = mechanism.bodies[4].state.vsol[2][3]
    ω11 = mechanism.bodies[1].state.ωsol[2][1]
    ω21 = mechanism.bodies[2].state.ωsol[2][1]
    ω31 = mechanism.bodies[3].state.ωsol[2][1]
    ω41 = mechanism.bodies[4].state.ωsol[2][1]
    return [v12, v13, v22, v23, v32, v33, v42, v43, ω11, ω21, ω31, ω41]
end
cache = [Vector{Float64}(), Vector{Float64}()]
for (id, yi) in enumerate(ytrain)
    kernel = SEArd(log.(params[2:end]), log(params[1]))
    mean = MeanDynamics(mechanism, _getμfb, id, cache)
    # mean = MeanZero()
    gp = GP(xtrain_old, yi, mean, kernel)
    # GaussianProcesses.optimize!(gp, LBFGS(linesearch = BackTracking(order=2)), Optim.Options(time_limit=10.))
    push!(gps, gp)
end

projectionerror = 0
getvω(μ) = return [SVector(0,μ[1:2]...), SVector(0,μ[3:4]...), SVector(0,μ[5:6]...), SVector(0,μ[7:8]...)], [SVector(μ[9],0,0), SVector(μ[10],0,0), SVector(μ[11],0,0), SVector(μ[12],0,0)]
suite[1] = @benchmark predictdynamics($mechanism, $gps, $xtest_old[1], $20, $getvω; regularizer = 1e-10)
display(suite[1])
# default: 170 ms
# no mean: 75 ms