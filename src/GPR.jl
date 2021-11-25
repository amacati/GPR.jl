module GPR

using StaticArrays
using LinearAlgebra
using Statistics
using Plots
using Optim
using ConstrainedDynamics
using ConstrainedDynamicsVis
using Rotations

export resetMechanism!
export updateMechanism!
export projectv!
export CState
export toState
export toStates

include("utils.jl")
include(joinpath("projections", "implicitProjection.jl"))
include("CState.jl")

end