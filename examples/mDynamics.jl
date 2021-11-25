using GaussianProcesses
using StaticArrays
using ConstrainedDynamics

include("utils.jl")
include("generatedata.jl")


struct MeanDynamics <: GaussianProcesses.Mean 
    mechanism::Mechanism
    getμ::Function
    xtransform::Function

    function MeanDynamics(mechanism::Mechanism, getμ; xtransform=(x, _) -> x)
        new(mechanism, getμ, xtransform)
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
    oldstates = getStates(mechanism)
    x = mDynamics.xtransform(x, mechanism)
    cstate = CState(x)
    setstates!(mechanism, cstate)
    newton!(mechanism)
    setstates!(mechanism, cstate)  # Only way to make results consistent. TODO: Ask Jan about this
    newton!(mechanism)
    μ = mDynamics.getμ(mechanism)
    for (id, state) in enumerate(oldstates)
        mechanism.bodies[id].state = state  # Reset mechanism to default values
    end
    return μ
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