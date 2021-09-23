using ConstrainedDynamics
using ConstrainedDynamics: GenericJoint,Vmat,params
using ConstrainedDynamicsVis
using StaticArrays

function simplependulum()
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
    storage = simulate!(mech,10.,record = true)
    return storage, mech
end


function ellipticjoint()
    # Parameters
    length1 = 1.0
    width, depth = 0.1, 0.1

    # Links
    origin = Origin{Float64}()
    link1 = Box(width, depth, length1, length1)


    @inline function g(joint::GenericJoint, xb::AbstractVector, qb::UnitQuaternion)
        a=5
        b=2

        if xb[3]==0
            xb[2] > 0 ? α = -pi/2 : α = pi/2
        elseif xb[3] > 0
            α = -atan(b^2/a^2*xb[2]/xb[3])
        else
            α = -atan(b^2/a^2*xb[2]/xb[3]) - pi
        end

        qα = UnitQuaternion(cos(α/2),sin(α/2),0,0,false)

        eqc1=Vmat(qb\qα)
        eqc2=SA[xb[1]; (xb[2]^2/a^2)+(xb[3]^2/b^2)-1]
        G= [eqc1;eqc2]
        return G
    end

    # Constraints
    joint_between_origin_and_link1 = EqualityConstraint(GenericJoint{5}(origin,link1,g))


    links = [link1]
    constraints = [joint_between_origin_and_link1]


    mech = Mechanism(origin, links, constraints)
    setPosition!(link1,x=[0;0.0;2],q = UnitQuaternion(RotX(0)))
    setVelocity!(link1,v=[0;2.0;0])

    steps = Base.OneTo(1000)
    storage = Storage{Float64}(steps,1)

    simulate!(mech, storage, record = true)
    return storage, mech
end

function doublependulum()
    # Parameters
    ex = [1.;0.;0.]

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

    storage = simulate!(mech, 10., record = true)
    return storage, mech
end