using GaussianProcesses
using StaticArrays
using ConstrainedDynamics

include("utils.jl")
include("generatedata.jl")


struct MeanDynamics <: GaussianProcesses.Mean 
    mechanism::Mechanism
    getμ::Function
    μID::Int
    cache
    xtransform::Function

    function MeanDynamics(mechanism::Mechanism, getμ, μID, cache; xtransform=(x, _) -> x)
        vcache = @view cache[:]
        new(mechanism, getμ, μID, vcache, xtransform)
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
    if mDynamics.cache[1] != x  # Cache is invalid
        mDynamics.cache[1] = x  # Set cache key
        mechanism = mDynamics.mechanism
        oldstates = getStates(mechanism)
        x = mDynamics.xtransform(x, mechanism)
        setstates!(mechanism, CState(x))
        newton!(mechanism)
        mDynamics.cache[2] = mDynamics.getμ(mechanism)  # Set cache data to result
        for (id, state) in enumerate(oldstates)
            mechanism.bodies[id].state = state  # Reset mechanism to default values
        end
    end
    return mDynamics.cache[2][mDynamics.μID]
end

function getμ(id::Integer)
    bodyid = div(id, 13) + 1
    stateid = (id - 7) % 13  # 3x, 4q values
    if stateid in 1:3
        return _getμv(mechanism) = mechanism.bodies[bodyid].state.vsol[2][stateid]
    elseif stateid in 4:6
        return _getμω(mechanism) = mechanism.bodies[bodyid].state.ωsol[2][stateid-3]  # -3 accounts for 3v before ω
    end
    throw(ArgumentError("Index $id does not point to v/ω!"))
end

#=
using GaussianProcesses
using StaticArrays
using ConstrainedDynamics

include("utils.jl")
include("generatedata.jl")


struct MeanDynamics <: GaussianProcesses.Mean 
    mechanism::Mechanism
    getμ::Function
    μID::Int
    cache::AbstractArray{AbstractArray}
    xtransform::Function

    function MeanDynamics(mechanism::Mechanism, getμ, μID; xtransform=(x, _) -> x)
        new(mechanism, getμ, μID, [Vector{Float64}(), Vector{Float64}()], xtransform)
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
    if mDynamics.cache[1] != x  # Cache is invalid
        mDynamics.cache[1] = x  # Set cache key
        mechanism = mDynamics.mechanism
        oldstates = getStates(mechanism)
        x = mDynamics.xtransform(x, mechanism)
        setstates!(mechanism, CState(x))
        newton!(mechanism)
        mDynamics.cache[2] = mDynamics.getμ(mechanism)  # Set cache data to result
        for (id, state) in enumerate(oldstates)
            mechanism.bodies[id].state = state  # Reset mechanism to default values
        end
    end
    return μ[mDynamics.μID]
end
=#