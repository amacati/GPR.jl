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

include(joinpath("projections", "implicitProjection.jl"))
end