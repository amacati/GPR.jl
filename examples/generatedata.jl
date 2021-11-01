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

function cleardata!(data; ϵ = 1e-2)
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
    deleteat!(data, sort!(collect(correlatedset)))
end

function max2mincoordinates(data, mechanism)
    mindata = Vector{SVector}()
    for maxstates in data
        states = [vector2state(maxstates[i:i+12]) for i in 1:13:length(maxstates)-12]
        resetMechanism!(mechanism, states)
        minstates = Vector{Float64}()
        for eqc in mechanism.eqconstraints
            append!(minstates, ConstrainedDynamics.minimalCoordinates(mechanism, eqc))
            append!(minstates, ConstrainedDynamics.minimalVelocities(mechanism, eqc))
        end
        push!(mindata, SVector(minstates...))
    end
    return mindata
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

function simplependulum2D(;Δt = 0.01, ω0 = SA[0;0;0], noise = false)
    joint_axis = [1.0; 0.0; 0.0]
    g = -9.81
    m = 1.0
    l = 1.0
    r = 0.01
    p2 = [0.0;0.0;l / 2] # joint connection point

    # Links
    origin = Origin{Float64}()
    link1 = Cylinder(r, l, m)

    # Constraints
    joint_between_origin_and_link1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=p2))
    links = [link1]
    constraints = [joint_between_origin_and_link1]

    mech = Mechanism(origin, links, constraints, g=g,Δt=Δt)
    
    noise ? ϵ = 0.1 : ϵ = 0
    q1 = UnitQuaternion(RotX(π / 2  + ϵ))
    setPosition!(origin,link1,p2 = p2,Δq = q1)
    setVelocity!(origin, link1; p2=p2, ω0)

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech,10.,record = true)
    return storage, mech, initialstates
end

function doublependulum2D(;Δt = 0.01, ω0 = [SA[0;0;0], SA[0;0;0]], noise = false)
    # Parameters
    l1 = 1.0
    l2 = sqrt(2) / 2
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]
    vert12 = -vert11
    vert21 = [0.;0.;l2 / 2]
    joint_axis1 = [1.0; 0.0; 0.0]
    joint_axis2 = [1.0; 0.0; 0.0]

    # Initial orientation
    noise ? ϵ = 0.1 : ϵ = 0
    phi1 = pi / 4 + ϵ
    q1 = UnitQuaternion(RotX(phi1))

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
    setPosition!(link1, link2, p1=vert12, p2=vert21, Δq=inv(q1)*UnitQuaternion(RotX(0.2)))
    setVelocity!(origin, link1; p2=vert11, Δω=ω0[1])
    setVelocity!(link1, link2; p1=vert12, p2=vert21, Δω=ω0[2])

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech, 10., record = true)
    return storage, mech, initialstates
end

function cartpole(;Δt = 0.01, x0=SA[0;0;0], q0=one(UnitQuaternion), v0 = SA[0;0;0], ω0 = SA[0;0;0])
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
    
    setPosition!(origin, link1; p2=p01, Δx=x0)
    setPosition!(link1, link2; p1=p12, p2=p21, Δq=q0)
    setVelocity!(origin, link1; p2=p01, Δv=v0)
    setVelocity!(link1, link2; p1=p12, p2=p21, Δω=ω0)

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech,10.,record = true)
    return storage, mech, initialstates
end

function simplependulum3D(; noise = false)
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

function rotatingcube()
    # Parameters
    l1 = 0.5
    x, y = .5, .5
    vert11 = [0.;0.;0]

    # Initial orientation
    phi1 = pi / 4
    q1 = UnitQuaternion(RotX(phi1))

    # Links
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))

    # Constraints
    socket0to1 = EqualityConstraint(Spherical(origin, link1; p2=vert11))
    
    links = [link1]
    constraints = [socket0to1]

    mech = Mechanism(origin, links, constraints)
    setPosition!(origin,link1,p2 = vert11,Δq = q1)
    setVelocity!(link1, ω=[1., 1., 1])
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech, 10., record = true)
    return storage, mech, initialstates
end
