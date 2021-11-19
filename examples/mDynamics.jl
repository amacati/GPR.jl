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
    oldstates = getstates(mechanism)
    x = mDynamics.xtransform(x, mechanism)
    setstates!(mechanism, tovstate(x))
    newton!(mechanism)
    μ = mDynamics.getμ(mechanism)
    for (id, state) in enumerate(oldstates)
        mechanism.bodies[id].state = state  # Reset mechanism to default values
    end
    return μ
end
