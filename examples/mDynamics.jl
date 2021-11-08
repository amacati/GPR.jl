using GaussianProcesses
using StaticArrays
using ConstrainedDynamics

include("utils.jl")
include("generatedata.jl")


struct MeanDynamics <: GaussianProcesses.Mean 
    mechanism::Mechanism
    bodyID::Integer
    entryID::Integer
    coords::String
    tfmin2max::Union{Function, Nothing}
    tfmax2min::Union{Function, Nothing}

    function MeanDynamics(mechanism::Mechanism, bodyID::Integer, entryID::Integer; coords::String = "max", 
                          tfmin2max = nothing, tfmax2min = nothing)
        new(mechanism, bodyID, entryID, coords, tfmin2max, tfmax2min)
    end 
end

GaussianProcesses.num_params(::MeanDynamics) = 0
GaussianProcesses.grad_mean(::MeanDynamics, ::AbstractVector) = Float64[]
GaussianProcesses.get_params(::MeanDynamics) = Float64[]
GaussianProcesses.get_param_names(::MeanDynamics) = Symbol[]

function GaussianProcesses.set_params!(::MeanDynamics, hyp::AbstractVector)
    length(hyp) == 0 || throw(ArgumentError("Mean dynamics function has no parameters"))
end

function GaussianProcesses.mean(mDynamics::MeanDynamics, x::AbstractVector)
    mechanism = mDynamics.mechanism
    oldstates = getstates(mechanism)
    mDynamics.coords == "min" && mDynamics.tfmin2max === nothing && (x = min2maxcoordinates(x, mechanism))
    mDynamics.coords == "min" && mDynamics.tfmin2max !== nothing && (x = mDynamics.tfmin2max(x))
    setstates!(mechanism, tovstate(x))
    newton!(mechanism)
    if mDynamics.coords == "min" && mDynamics.tfmax2min !== nothing
        μ = tfmax2min(mechanism)
    elseif mDynamics.entryID < 4  # v -> 1 2 3, ω -> 4, 5, 6
        μ = mechanism.bodies[mDynamics.bodyID].state.vsol[2][mDynamics.entryID]
    else
        μ = mechanism.bodies[mDynamics.bodyID].state.ωsol[2][mDynamics.entryID-3]
    end
    for (id, state) in enumerate(oldstates)
        mechanism.bodies[id].state = state  # Reset mechanism to default values
    end
    return μ
end
