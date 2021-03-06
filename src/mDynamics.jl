using GaussianProcesses
using StaticArrays
using ConstrainedDynamics


mutable struct MDCache
    key::AbstractArray
    data::AbstractArray

    MDCache() = new(Vector{Float64}(), Vector{Float64}())
end

struct MeanDynamics <: GaussianProcesses.Mean 
    mechanism::Mechanism
    getμ::Function
    μID::Int
    cache::MDCache
    xtransform::Function

    function MeanDynamics(mechanism::Mechanism, getμ::Function, μID::Int, cache::MDCache; xtransform=(x, _) -> x)
        new(mechanism, getμ, μID, cache, xtransform)
    end

    function MeanDynamics(mechanism::Mechanism, getμ::Function, μID::Int; xtransform=(x, _) -> x)
        new(mechanism, getμ, μID, MDCache(), xtransform)
    end 
end

GaussianProcesses.num_params(::MeanDynamics) = 0
GaussianProcesses.grad_mean(::MeanDynamics, ::AbstractVector) = Float64[]
GaussianProcesses.get_params(::MeanDynamics) = Float64[]
GaussianProcesses.get_param_names(::MeanDynamics) = Symbol[]

function GaussianProcesses.set_params!(::MeanDynamics, hyp::AbstractVector)
    length(hyp) == 0 || throw(ArgumentError("Mean dynamics function has no parameters"))
end

"""
    Mean dynamics for the GPs. Use a cache to avoid recalculating the dynamics of the same training samples over and over again.
"""
function GaussianProcesses.mean(mDynamics::MeanDynamics, x::AbstractVector)
    if mDynamics.cache.key != x  # Cache is invalid
        mDynamics.cache.key = x  # Set cache to current input state
        mechanism = mDynamics.mechanism
        oldstates = getStates(mechanism)
        x = mDynamics.xtransform(x, mechanism)
        setstates!(mechanism, CState(x))
        ConstrainedDynamics.newton!(mechanism)
        mDynamics.cache.data = mDynamics.getμ(mechanism)  # Set cache data to result
        for (id, state) in enumerate(oldstates)
             mechanism.bodies[id].state = state  # Reset mechanism to default values
        end
    end
    return mDynamics.cache.data[mDynamics.μID]
end

function getμ(ids::AbstractArray{Int,1})
    _getμ(mech) = return CState(mech, usesolution=true)[ids]
    return _getμ
end
