using GaussianProcesses
using StaticArrays
using ConstrainedDynamics

include("utils.jl")


struct MeanDynamics <: GaussianProcesses.Mean 
    mechanism::Mechanism
    bodyID::Integer
    entryID::Integer

    MeanDynamics(mechanism::Mechanism, bodyID::Integer, entryID::Integer) = new(mechanism, bodyID, entryID)
end

GaussianProcesses.num_params(::MeanDynamics) = 0
GaussianProcesses.grad_mean(::MeanDynamics, ::AbstractVector) = Float64[]
GaussianProcesses.get_params(::MeanDynamics) = Float64[]
GaussianProcesses.get_param_names(::MeanDynamics) = Symbol[]

function GaussianProcesses.set_params!(::MeanDynamics, hyp::AbstractVector)
    length(hyp) == 0 || throw(ArgumentError("Zero mean function has no parameters"))
end

function GaussianProcesses.mean(mDynamics::MeanDynamics, x::AbstractVector)
    oldstates = getstates(mDynamics.mechanism)
    setstates!(mechanism, tovstate(x))
    newton!(mechanism)
    if mDynamics.entryID < 4  # v -> 1 2 3, ω -> 4, 5, 6
        μ = mechanism.bodies[mDynamics.bodyID].state.vsol[2][mDynamics.entryID]
    else
        μ = mechanism.bodies[mDynamics.bodyID].state.ωsol[2][mDynamics.entryID-3]
    end
    for (id, state) in enumerate(oldstates)
        mechanism.bodies[id].state = state  # Reset mechanism to default values
    end
    return μ
end
