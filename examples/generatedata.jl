using ConstrainedDynamics
using ConstrainedDynamicsVis
using StaticArrays
using Rotations

function loaddata(storage)
    Nbodies = length(storage.x)
    Nsamples = length(storage.x[1])
    X = Vector{SVector{Nbodies*13, Float64}}()
    for t = 1:Nsamples
        sample = [[storage.x[id][t]..., storage.q[id][t].w, storage.q[id][t].x, storage.q[id][t].y, storage.q[id][t].z, storage.v[id][t]..., storage.ω[id][t]...]
                   for id in 1:Nbodies]
        sample = reduce(vcat, sample)
        push!(X, sample)
    end
    return deepcopy(X)
end

function simplependulum2D()
    joint_axis = [1.0; 0.0; 0.0]

    Δt=0.01
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

    q1 = UnitQuaternion(RotX(π / 2))
    setPosition!(origin,link1,p2 = p2,Δq = q1)
    setVelocity!(link1)

    initialstates = [deepcopy(body.state) for body in mech.bodies]
    storage = simulate!(mech,10.,record = true)
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
    phi1 = pi / 4
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

function simplependulum3D()
    # Parameters
    l1 = 1.0
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]

    # Initial orientation
    phi1 = pi / 4
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
