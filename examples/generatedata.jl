using ConstrainedDynamics
using ConstrainedDynamicsVis
using StaticArrays
using Rotations


function loaddata(storage; coordinates="maximal", mechanism = nothing)
    Nbodies = length(storage.x)
    Nsamples = length(storage.x[1])
    X = Vector{SVector{Nbodies*13, Float64}}()
    for t = 1:Nsamples
        sample = [[storage.x[id][t]..., storage.q[id][t].w, storage.q[id][t].x, storage.q[id][t].y, storage.q[id][t].z, storage.v[id][t]..., storage.ω[id][t]...]
                   for id in 1:Nbodies]
        sample = reduce(vcat, sample)
        push!(X, sample)
    end
    coordinates == "maximal" && return X
    mechanism === nothing && throw(ArgumentError("Loading data in non-maximal coordinates needs a mechanism argument!"))
    coordinates == "minimal" && return max2mincoordinates(X, mechanism)
    throw(ArgumentError("Coordinates setting $coordinates not supported!"))
end

function _correlated_indices(data::AbstractArray{<:AbstractArray{Float64}}, ϵ)
    N = length(data[1])
    correlatedset = Set{Int}()
    for i in 1:length(data)
        if i in correlatedset
            continue
        end
        for j in i+1:length(data)
            if sum((data[i]-data[j]).^2/N) < ϵ
                push!(correlatedset, j)
            end
        end
    end
    return sort!(collect(correlatedset))
end

function cleardata!(data::AbstractArray{<:AbstractArray{Float64}}; ϵ = 1e-2)
    correlated_indices = _correlated_indices(data, ϵ)
    deleteat!(data, correlated_indices)
end

function cleardata!(datacollection::Tuple; ϵ = 1e-2)
    correlated_indices = _correlated_indices(datacollection[1], ϵ)
    for data in datacollection
        deleteat!(data, correlated_indices)
    end
end

function max2mincoordinates(cstate::Vector{Float64}, mechanism)
    states = tostates(cstate)
    resetMechanism!(mechanism, states)
    cstate_min = Vector{Float64}()
    for eqc in mechanism.eqconstraints
        append!(cstate_min, ConstrainedDynamics.minimalCoordinates(mechanism, eqc))
        append!(cstate_min, ConstrainedDynamics.minimalVelocities(mechanism, eqc))
    end
    return cstate_min
end

function min2maxcoordinates(data, mechanism)
    maxdata = Vector{Vector{Float64}}()
    N = 0
    for minstates in data
        for eqc in mechanism.eqconstraints
            Nc = length(eqc.constraints)
            ConstrainedDynamics.setPosition!(mechanism, eqc, [minstates[N+1:N+Nc]])
            ConstrainedDynamics.setVelocity!(mechanism, eqc, [minstates[N+Nc+1:N+2Nc]])
            N += 2Nc
        end
        push!(maxdata, SVector(vcat(getstates(mechanism)...)))
    end
    return maxdata
end

function simplependulum2D(;Δt = 0.01, θstart = 0., ωstart = 0.)
    joint_axis = [1.0; 0.0; 0.0]
    g = -9.81
    m = 1.0
    l = 1.0
    r = 0.01
    p2 = [0.0;0.0;l / 2] # joint connection point
    ΔT = 3.  # Simulation duration

    # Links
    origin = Origin{Float64}()
    link1 = Cylinder(r, l, m)

    # Constraints
    joint_between_origin_and_link1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=p2))
    links = [link1]
    constraints = [joint_between_origin_and_link1]

    mech = Mechanism(origin, links, constraints, g=g, Δt=Δt)
    
    q1 = UnitQuaternion(RotX(θstart))
    setPosition!(origin, link1; p2=p2, Δq=q1)
    setVelocity!(origin, link1; p2=p2, Δω=SA[ωstart, 0., 0.])

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech, ΔT, record = true)
    return storage, mech, initialstates
end

function doublependulum2D(;Δt = 0.01, θstart = [0., 0.], ωstart = [0., 0.])
    # Parameters
    l1 = 1.0
    l2 = sqrt(2) / 2
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]
    vert12 = -vert11
    vert21 = [0.;0.;l2 / 2]
    joint_axis1 = [1.0; 0.0; 0.0]
    joint_axis2 = [1.0; 0.0; 0.0]
    ΔT = 3.

    # Initial orientation
    q1 = UnitQuaternion(RotX(θstart[1]))
    q2 = UnitQuaternion(RotX(θstart[2]))

    # Links
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
    link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

    # Constraints
    socket0to1 = EqualityConstraint(Revolute(origin, link1, joint_axis1; p2=vert11))
    socket1to2 = EqualityConstraint(Revolute(link1, link2, joint_axis2; p1=vert12, p2=vert21))
    links = [link1;link2]
    constraints = [socket0to1;socket1to2]

    mech = Mechanism(origin, links, constraints, Δt=Δt)

    setPosition!(origin,link1, p2=vert11, Δq=q1)
    setPosition!(link1, link2, p1=vert12, p2=vert21, Δq=inv(q1)*q2)
    setVelocity!(origin, link1; p2=vert11, Δω=SA[ωstart[1], 0., 0.])
    setVelocity!(link1, link2; p1=vert12, p2=vert21, Δω=SA[ωstart[2], 0., 0.])

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech, ΔT, record = true)
    return storage, mech, initialstates
end

function cartpole(;Δt = 0.01, xstart=0., θstart=0., vstart = 0., ωstart = 0.)
    xaxis = [1.0; 0.0; 0.0]
    yaxis = [0.0; 1.0; 0.0]
    g = -9.81
    m1 = 1.0
    m2 = 1.0
    l = 0.5
    r = 0.01
    p01 = [0.0; 0.0; 0.0] # joint connection point
    p12 = [0.0; 0.0; 0.0]
    p21 = [0.0; 0.0; l/2]
    ΔT = 3.

    # Links
    x, y, z = 0.2, 0.3, 0.1
    origin = Origin{Float64}()
    link1 = Box(x, y, z, m1)
    link2 = Cylinder(r, l, m2)

    # Constraints
    joint_origin_link1 = EqualityConstraint(Prismatic(origin, link1, yaxis; p2=p01))
    joint_link1_link2 = EqualityConstraint(Revolute(link1, link2, xaxis; p1=p12, p2=p21))
    links = [link1, link2]
    constraints = [joint_origin_link1, joint_link1_link2]

    mech = Mechanism(origin, links, constraints, g=g, Δt=Δt)
    
    setPosition!(origin, link1; p2=p01, Δx=SA[0., xstart, 0.])
    setPosition!(link1, link2; p1=p12, p2=p21, Δq=UnitQuaternion(RotX(θstart)))
    setVelocity!(origin, link1; p2=p01, Δv=SA[0., vstart, 0.])
    setVelocity!(link1, link2; p1=p12, p2=p21, Δω=SA[ωstart, 0., 0.])

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech, ΔT, record = true)
    return storage, mech, initialstates
end

function simplependulum3D()
    # Parameters
    l1 = 1.0
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]

    # Initial orientation
    noise ? ϵ = 0.1 : ϵ = 0
    phi1 = pi / 4 + ϵ
    q1 = UnitQuaternion(RotX(phi1))

    # Links
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
    # link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

    # Constraints
    socket0to1 = EqualityConstraint(Spherical(origin, link1; p2=vert11))
    
    links = [link1]
    constraints = [socket0to1]

    mech = Mechanism(origin, links, constraints)
    setPosition!(origin,link1,p2 = vert11,Δq = q1)
    setVelocity!(link1, v=[1., 0., 0])

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech, 10., record = true)
    return storage, mech, initialstates
end

function doublependulum3D()
    # Parameters
    l1 = 1.0
    l2 = sqrt(2) / 2
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]
    vert12 = -vert11
    vert21 = [0.;0.;l2 / 2]

    # Initial orientation
    noise ? ϵ = 0.1 : ϵ = 0
    phi1 = pi / 4 + ϵ
    q1 = UnitQuaternion(RotX(phi1))

    # Links
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
    link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

    # Constraints
    socket0to1 = EqualityConstraint(Spherical(origin, link1; p2=vert11))
    socket1to2 = EqualityConstraint(Spherical(link1, link2; p1=vert12, p2=vert21))

    links = [link1;link2]
    constraints = [socket0to1;socket1to2]
    mech = Mechanism(origin, links, constraints)
    setPosition!(origin,link1,p2 = vert11,Δq = q1)
    setPosition!(link1,link2,p1 = vert12,p2 = vert21,Δq = inv(q1)*UnitQuaternion(RotY(0.2)))

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech, 10., record = true)
    return storage, mech, initialstates
end
