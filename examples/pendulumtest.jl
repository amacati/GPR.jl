using GPR
using ConstrainedDynamicsVis

include("myNewton.jl")


function myNewton!(mechanism)
    Nbodies = length(mechanism.bodies)
    Ndims = sum([length(eqc) for eqc in mechanism.eqconstraints])  # Total dimensionality of constraints
    Δsinit = zeros(MVector{6*Nbodies + Ndims})
    for (id, body) in enumerate(mechanism.bodies)
        offset = (id-1)*6
        myDynamics!(mechanism, body, offset, Δsinit)
    end
    _myNewton!(mechanism, Δsinit, ϵ = 1e-10, newtonIter = 100)
end

function mySimulate!(mechanism, t; record = false)
    steps = Int(t/mechanism.Δt)
    record ? storage = Storage{Float64}(steps,length(mechanism.bodies)) : storage = Storage{Float64}()
    foreach(ConstrainedDynamics.setsolution!, mechanism.bodies)
    for i in 1:steps
        myNewton!(mechanism)
        foreach(ConstrainedDynamics.updatesolution!, mechanism.bodies)
        foreach(ConstrainedDynamics.updatesolution!, mechanism.eqconstraints)
        record && ConstrainedDynamics.saveToStorage!(mechanism, storage, i)
    end
    record ? (return storage) : (return) 
end

# Comparison for solution
storage, mechanism, initialstates = doublependulum3D()
resetMechanism!(mechanism, initialstates, overwritesolution = true)  # Reset mechanism to starting position
storage = mySimulate!(mechanism, 10., record = true)
ConstrainedDynamicsVis.visualize(mechanism, storage; showframes = true, env = "editor")