using GaussianProcesses
using GPR
using Optim
using ConstrainedDynamicsVis
using ConstrainedDynamics
using LineSearches
using Statistics

include(joinpath("..", "generatedata.jl"))


storage, mechanism, initialstates = cartpole(Δt=0.01, q0=UnitQuaternion(RotX(π/2)))
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")
