using ConstrainedDynamics
using ConstrainedDynamicsVis
using StaticArrays
using Rotations
using DataFrames


function max2mincoordinates(cstate::Vector{Float64}, mechanism)
    oldstates = getstates(mechanism)
    states = tostates(cstate)
    resetMechanism!(mechanism, states)
    cstate_min = Vector{Float64}()
    for eqc in mechanism.eqconstraints
        append!(cstate_min, ConstrainedDynamics.minimalCoordinates(mechanism, eqc))
        append!(cstate_min, ConstrainedDynamics.minimalVelocities(mechanism, eqc))
    end
    for (id, state) in enumerate(oldstates)
        mechanism.bodies[id].state = state  # Reset mechanism to default values
    end
    return cstate_min
end

function min2maxcoordinates(cstate::AbstractArray, mechanism::Mechanism)
    oldstates = getstates(mechanism)
    maxdata = Vector{Float64}(undef, 13*length(mechanism.bodies))
    N = 0
    for eqc in mechanism.eqconstraints
        Nc = 6 - sum([length(c) for c in eqc.constraints])
        ConstrainedDynamics.setPosition!(mechanism, eqc, SVector(cstate[N+1:N+Nc]...))
        ConstrainedDynamics.setVelocity!(mechanism, eqc, SVector(cstate[N+Nc+1:N+2Nc]...))
        N += 2Nc
    end
    for (id, body) in enumerate(mechanism.bodies)
        offset = (id-1)*13
        maxdata[1+offset:3+offset] = body.state.xc
        q = body.state.qc
        maxdata[4+offset:7+offset] = [q.w, q.x, q.y, q.z]
        maxdata[8+offset:10+offset] = body.state.vc
        maxdata[11+offset:13+offset] = body.state.ωc
    end
    for (id, state) in enumerate(oldstates)
        mechanism.bodies[id].state = state  # Reset mechanism to default values
    end
    return maxdata
end

function simplependulum2D(steps; Δt = 0.01, θstart = 0., ωstart = 0., m = 1.0, ΔJ = SMatrix{3,3,Float64}(zeros(9)...), friction = 0, threadlock = nothing)
    joint_axis = [1.0; 0.0; 0.0]
    g = -9.81
    l = 1.0
    r = 0.01
    p2 = [0.0;0.0;l / 2] # joint connection point
    ΔT = Δt * steps
    if threadlock === nothing
        mech, origin, link1 = _simplependulum2Dmech(r, l, m, ΔJ, joint_axis, p2, g, Δt)
    else
        lock(threadlock)  # ConstrainedDynamics uses globals with possible data races to initialize mechanisms -> lock
        try
            mech, origin, link1 = _simplependulum2Dmech(r, l, m, ΔJ, joint_axis, p2, g, Δt)
        finally
            unlock(threadlock)
        end
    end
    q1 = UnitQuaternion(RotX(θstart))
    setPosition!(origin, link1; p2=p2, Δq=q1)
    setVelocity!(origin, link1; p2=p2, Δω=SA[ωstart, 0., 0.])
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if friction != 0
        function control!(mech, _)
            setForce!(mech.bodies[1], τ=-SA[1.,0,0]friction*mech.bodies[1].state.ωc[1])
        end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end    
    return storage, mech, initialstates
end

function _simplependulum2Dmech(r, l, m, ΔJ, joint_axis, p2, g, Δt)
    origin = Origin{Float64}()
    link1 = Cylinder(r, l, m)
    link1.J = abs.(link1.J + ΔJ)
    joint_between_origin_and_link1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=p2))
    links = [link1]
    constraints = [joint_between_origin_and_link1]
    return Mechanism(origin, links, constraints, g=g, Δt=Δt), origin, link1
end

function doublependulum2D(steps; Δt = 0.01, θstart = [0., 0.], ωstart = [0., 0.], m = [1., sqrt(2)/2],
                           ΔJ = [SMatrix{3,3,Float64}(zeros(9)...), SMatrix{3,3,Float64}(zeros(9)...)], friction = [0,0], threadlock = nothing)
    # Parameters
    l1 = 1.0
    l2 = sqrt(2) / 2
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]
    vert12 = -vert11
    vert21 = [0.;0.;l2 / 2]
    joint_axis = [1.0; 0.0; 0.0]
    ΔT = Δt * steps
    # Initial orientation
    q1 = UnitQuaternion(RotX(θstart[1]))
    q2 = UnitQuaternion(RotX(θstart[2]))
    if threadlock === nothing
        mech, origin, link1, link2 = _doublependulum2Dmech(x, y, l1, l2, m, ΔJ, joint_axis, vert11, vert12, vert21, Δt)
    else
        lock(threadlock)
        try
            mech, origin, link1, link2 = _doublependulum2Dmech(x, y, l1, l2, m, ΔJ, joint_axis, vert11, vert12, vert21, Δt)
        finally
            unlock(threadlock)
        end
    end
    setPosition!(origin,link1, p2=vert11, Δq=q1)
    setPosition!(link1, link2, p1=vert12, p2=vert21, Δq=q2)
    setVelocity!(origin, link1; p2=vert11, Δω=SA[ωstart[1], 0., 0.])
    setVelocity!(link1, link2; p1=vert12, p2=vert21, Δω=SA[ωstart[2], 0., 0.])
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if (any(friction .!= 0))
        function control!(mech, _)
            setForce!(mech.bodies[1], τ=-SA[1.,0,0]friction[1]*mech.bodies[1].state.ωc[1])
            setForce!(mech.bodies[2], τ=-SA[1.,0,0]friction[2]*mech.bodies[2].state.ωc[1])
        end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end
    return storage, mech, initialstates
end

function _doublependulum2Dmech(x, y, l1, l2, m, ΔJ, joint_axis, vert11, vert12, vert21, Δt)
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, m[1], color = RGBA(1., 1., 0.))
    link2 = Box(x, y, l2, m[2], color = RGBA(1., 1., 0.))
    link1.J = abs.(link1.J + ΔJ[1])
    link2.J = abs.(link2.J + ΔJ[2])
    socket0to1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=vert11))
    socket1to2 = EqualityConstraint(Revolute(link1, link2, joint_axis; p1=vert12, p2=vert21))
    links = [link1;link2]
    constraints = [socket0to1;socket1to2]
    mech = Mechanism(origin, links, constraints, Δt=Δt)
    return mech, origin, link1, link2
end

function cartpole(steps; Δt = 0.01, xstart=0., θstart=0., vstart = 0., ωstart = 0., m = [1., 1.],
                   ΔJ = [SMatrix{3,3,Float64}(zeros(9)...), SMatrix{3,3,Float64}(zeros(9)...)], friction = [0, 0], threadlock = nothing)
    xaxis = [1.0; 0.0; 0.0]
    yaxis = [0.0; 1.0; 0.0]
    g = -9.81
    l = 0.5
    r = 0.01
    p01 = [0.0; 0.0; 0.0] # joint connection point
    p12 = [0.0; 0.0; 0.0]
    p21 = [0.0; 0.0; l/2]
    ΔT = steps*Δt
    x, y, z = 0.2, 0.3, 0.1
    if threadlock === nothing
        mech, origin, link1, link2 = _cartpolemech(x, y, z, r, l, m, ΔJ, xaxis, yaxis, p01, p12, p21, Δt)
    else
        lock(threadlock)
        try
            mech, origin, link1, link2 = _cartpolemech(x, y, z, r, l, m, ΔJ, xaxis, yaxis, p01, p12, p21, Δt)     
        finally
            unlock(threadlock)
        end
    end
    setPosition!(origin, link1; p2=p01, Δx=SA[0., xstart, 0.])
    setPosition!(link1, link2; p1=p12, p2=p21, Δq=UnitQuaternion(RotX(θstart)))
    setVelocity!(origin, link1; p2=p01, Δv=SA[0., vstart, 0.])
    setVelocity!(link1, link2; p1=p12, p2=p21, Δω=SA[ωstart, 0., 0.])
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if (any(friction .!= 0))
        function control!(mech, _)
            setForce!(mech.bodies[1], F=-SA[0,1.,0]friction[1]*mech.bodies[1].state.vc[2])
            setForce!(mech.bodies[2], τ=-SA[1.,0,0]friction[2]*mech.bodies[2].state.ωc[1])
        end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end
    return storage, mech, initialstates
end

function _cartpolemech(x, y, z, r, l, m, ΔJ, xaxis, yaxis, p01, p12, p21, Δt)
    origin = Origin{Float64}()
    link1 = Box(x, y, z, m[1])
    link2 = Cylinder(r, l, m[2])
    link1.J = abs.(link1.J + ΔJ[1])
    link2.J = abs.(link2.J + ΔJ[2])
    joint_origin_link1 = EqualityConstraint(Prismatic(origin, link1, yaxis; p2=p01))
    joint_link1_link2 = EqualityConstraint(Revolute(link1, link2, xaxis; p1=p12, p2=p21))
    links = [link1, link2]
    constraints = [joint_origin_link1, joint_link1_link2]
    mech = Mechanism(origin, links, constraints, Δt=Δt)
    return mech, origin, link1, link2
end

function fourbar(steps; Δt = 0.01, θstart = [0., 0.], m = 1., ΔJ = SMatrix{3,3,Float64}(zeros(9)...), friction = [0,0], threadlock = nothing)
    # Parameters
    ex = [1.;0.;0.]
    l = 1.0
    x, y = .1, .1
    vert11 = [0.;0.;l / 2]
    vert12 = -vert11
    ΔT = steps*Δt
    # Initial orientation
    q = UnitQuaternion(RotX(θstart[2]))
    qoff = UnitQuaternion(RotX(θstart[1]))
    Δq1 = q * qoff
    Δq2 = q

    if threadlock === nothing
        mech, origin, links = _fourbarmech(x, y, l, m, ΔJ, ex, vert11, vert12, Δt)
    else
        lock(threadlock)
        try
            mech, origin, links = _fourbarmech(x, y, l, m, ΔJ, ex, vert11, vert12, Δt)
        finally
            unlock(threadlock)
        end
    end
    setPosition!(origin, links[1], p2 = vert11, Δq = Δq1)
    setPosition!(links[1], links[2], p1 = vert12, p2 = vert11, Δq = inv(Δq2) * inv(Δq2))
    setPosition!(links[1], links[3], p1 = vert11, p2 = vert11, Δq = inv(Δq2) * inv(Δq2))
    setPosition!(links[3], links[4], p1 = vert12, p2 = vert11, Δq = Δq2 * Δq2)
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if (any(friction .!= 0))
        function control!(mech, _)
            setForce!(mech.bodies[1], F=-SA[1.,0,0]friction[1]*mech.bodies[1].state.ωc[1])
            setForce!(mech.bodies[3], τ=-SA[1.,0,0]friction[2]*mech.bodies[3].state.ωc[1])
        end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end
    return storage, mech, initialstates
end

function _fourbarmech(x, y, l, m, ΔJ, ex, vert11, vert12, Δt)
    origin = Origin{Float64}()
    links = [Box(x, y, l, m, color = RGBA(1., 1., 0.)) for _ in 1:4]
    for link in links
        link.J = abs.(link.J + ΔJ)  # Inertia modification
    end
    j1 = EqualityConstraint(Revolute(origin, links[1], ex; p2=vert11))
    j2 = EqualityConstraint(Revolute(links[1], links[2], ex; p1=vert12, p2=vert11), Cylindrical(links[1], links[3], ex; p1=vert11, p2=vert11))
    j3 = EqualityConstraint(Revolute(links[3], links[4], ex; p1=vert12, p2=vert11))
    j4 = EqualityConstraint(Revolute(links[2], links[4], ex; p1=vert12, p2=vert12))
    constraints = [j1, j2, j3, j4]
    mech = Mechanism(origin, links, constraints, Δt=Δt)
    return mech, origin, links
end

function generate_dataframes(config, nsamples, exp1, exp2, exptest; parallel = false)
    parallel && return _generate_dataframes_p(config, nsamples, exp1, exp2, exptest)
    return _generate_dataframes(config, nsamples, exp1, exp2, exptest)
end

function _generate_dataframes(config, nsamples, exp1, exp2, exptest)
    scaling = Int(0.01/config["Δtsim"])
    traindf = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}())
    for _ in 1:div(nsamples, 2)
        storage = exp1()  # Simulate 2 secs from random position, choose one sample
        j = rand(1:2*Int(1/config["Δtsim"]) - scaling)  # End of storage - required steps
        push!(traindf, (getstates(storage, j), getstates(storage, j+scaling)))
    end
    for _ in 1:div(nsamples, 2)
        storage = exp2()
        j = rand(1:2*Int(1/config["Δtsim"]) - scaling)  # End of storage - required steps
        push!(traindf, (getstates(storage, j), getstates(storage, j+scaling)))
    end
    testdf = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}(), sfuture = Vector{Vector{State}}())
    for _ in 1:config["testsamples"]
        storage = exptest()  # Simulate 2 secs from random position, choose one sample
        j = rand(1:2*Int(1/config["Δtsim"]) - scaling*(config["simsteps"]+1))  # End of storage - required steps
        push!(testdf, (getstates(storage, j), getstates(storage, j+scaling), getstates(storage, j+scaling*(config["simsteps"]+1))))
    end
    return traindf, testdf
end

function _generate_dataframes_p(config, nsamples, exp1, exp2, exptest)
    scaling = Int(0.01/config["Δtsim"])
    traindf = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}())
    threadlock = ReentrantLock()  # Push to df not atomic
    Threads.@threads for _ in 1:div(nsamples, 2)
        storage = exp1()  # Simulate 2 secs from random position, choose one sample
        j = rand(1:2*Int(1/config["Δtsim"]) - scaling)  # End of storage - required steps
        lock(threadlock)
        try
            push!(traindf, (getstates(storage, j), getstates(storage, j+scaling)))
        finally
            unlock(threadlock)
        end
    end
    Threads.@threads for _ in 1:div(nsamples, 2)
        storage = exp2()
        j = rand(1:2*Int(1/config["Δtsim"]) - scaling)  # End of storage - required steps
        lock(threadlock)
        try
            push!(traindf, (getstates(storage, j), getstates(storage, j+scaling)))
        finally
            unlock(threadlock)
        end
    end
    testdf = DataFrame(sold = Vector{Vector{State}}(), scurr = Vector{Vector{State}}(), sfuture = Vector{Vector{State}}())
    Threads.@threads for _ in 1:config["testsamples"]
        storage = exptest()  # Simulate 2 secs from random position, choose one sample
        j = rand(1:2*Int(1/config["Δtsim"]) - scaling*(config["simsteps"]+1))  # End of storage - required steps
        lock(threadlock)
        try
            push!(testdf, (getstates(storage, j), getstates(storage, j+scaling), getstates(storage, j+scaling*(config["simsteps"]+1))))
        finally
            unlock(threadlock)
        end
    end
    return traindf, testdf
end

"""
    Noise for v is calculated by doing a forward integration of ω in minimal coordinates, finding x2 and building a v that is consistent
    with the variational integration approach.
"""
function applynoise!(df, Σ, etype, Δtsim, varargs...)
    etype == "P1" && return _applynoise_p1!(df, Σ, Δtsim, varargs...)
    etype == "P2" && return _applynoise_p2!(df, Σ, Δtsim, varargs...)
    etype == "CP" && return _applynoise_cp!(df, Σ, Δtsim, varargs...)
    etype == "FB" && return _applynoise_fb!(df, Σ, Δtsim, varargs...)
    throw(ArgumentError("Experiment $etype is not supported!"))
end

function _applynoise_p1!(df, Σ, Δtsim, l)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][1].qc  # Small error around θ
            col[t][1].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ = Rotations.rotation_angle(col[t][1].qc)*sign(col[t][1].qc.x)*sign(col[t][1].qc.w)  # Signum for axis direction
            ω = col[t][1].ωc[1]
            col[t][1].xc = [0, l/2*sin(θ), -l/2*cos(θ)]  # Noise is consequence of θ and ω
            θnext = ω*Δtsim + θ
            xnext = [0, l/2*sin(θnext), -l/2*cos(θnext)]
            v = (xnext - col[t][1].xc)/Δtsim
            col[t][1].vc = v
        end
    end
end

function _applynoise_p2!(df, Σ, Δtsim, l1, l2)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][1].qc
            col[t][1].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            col[t][2].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][2].qc
            col[t][2].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ1 = Rotations.rotation_angle(col[t][1].qc)*sign(col[t][1].qc.x)*sign(col[t][1].qc.w)  # Signum for axis direction
            θ2 = Rotations.rotation_angle(col[t][2].qc)*sign(col[t][2].qc.x)*sign(col[t][2].qc.w) - θ1
            ω1, ω2 = col[t][1].ωc[1], col[t][2].ωc[1] - col[t][1].ωc[1]
            col[t][1].xc = [0, l1/2*sin(θ1), -l1/2*cos(θ1)]  # Noise is consequence of θ and ω
            col[t][2].xc = [0, l1*sin(θ1) + l2/2*sin(θ1+θ2), -l1*cos(θ1) - l2/2*cos(θ1+θ2)]  # Noise is consequence of θ and ω
            θ1next = θ1 + ω1*Δtsim
            θ2next = θ2 + ω2*Δtsim
            x1next = [0, l1/2*sin(θ1next), -l1/2*cos(θ1next)]
            x2next = [0, l1*sin(θ1next) + l2/2*sin(θ1next+θ2next), -l1*cos(θ1next) - l2/2*cos(θ1next+θ2next)]
            v1 = (x1next - col[t][1].xc) / Δtsim
            v2 = (x2next - col[t][2].xc) / Δtsim
            col[t][1].vc = v1
            col[t][2].vc = v2
        end
    end
end

function _applynoise_cp!(df, Σ, Δtsim, l)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].xc += Σ["x"]*[0, randn(), 0]  # Cart pos noise only y, no orientation noise
            col[t][1].vc += Σ["v"]*[0, randn(), 0]  # Same for v
            col[t][2].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][2].qc
            col[t][2].ωc += Σ["ω"]*[randn(), 0, 0]
            θ = Rotations.rotation_angle(col[t][2].qc)*sign(col[t][2].qc.x)*sign(col[t][2].qc.w)  # Signum for axis direction
            ω = col[t][2].ωc[1]
            col[t][2].xc = col[t][1].xc + [0, l/2*sin(θ), -l/2*cos(θ)]
            θnext = ω*Δtsim + θ
            xnext = col[t][1].xc + col[t][1].vc*Δtsim + [0, l/2*sin(θnext), -l/2*cos(θnext)]
            v = (xnext - col[t][2].xc)/Δtsim
            col[t][2].vc = v
        end
    end
end

function _applynoise_fb!(df, Σ, Δtsim, l)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][1].qc
            col[t][1].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            col[t][3].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][3].qc
            col[t][3].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ1 = Rotations.rotation_angle(col[t][1].qc)*sign(col[t][1].qc.x)*sign(col[t][1].qc.w)  # Signum for axis direction
            θ2 = Rotations.rotation_angle(col[t][3].qc)*sign(col[t][3].qc.x)*sign(col[t][3].qc.w)
            ω1, ω2 = col[t][1].ωc[1], col[t][3].ωc[1]
            col[t][2].qc = UnitQuaternion(RotX(θ2))
            col[t][4].qc = UnitQuaternion(RotX(θ1))
            col[t][1].xc = [0, 0.5sin(θ1)l, -0.5cos(θ1)l]
            col[t][2].xc = [0, sin(θ1)l + 0.5sin(θ2)l, -cos(θ1)l - 0.5cos(θ2)l]
            col[t][3].xc = [0, 0.5sin(θ2)l, -0.5cos(θ2)l]
            col[t][4].xc = [0, sin(θ2)l + 0.5sin(θ1)l, -cos(θ2)l - 0.5cos(θ1)l]
            θ1next = ω1*Δtsim + θ1
            θ2next = ω2*Δtsim + θ2
            x1next = [0, 0.5sin(θ1next)l, -0.5cos(θ1next)l]
            x2next = [0, sin(θ1next)l + 0.5sin(θ2next)l, -cos(θ1next)l - 0.5cos(θ2next)l]
            x3next = [0, 0.5sin(θ2next)l, -0.5cos(θ2next)l]
            x4next = [0, sin(θ2next)l + 0.5sin(θ1next)l, -cos(θ2next)l - 0.5cos(θ1next)l]
            v1 = (x1next - col[t][1].xc)/Δtsim
            v2 = (x2next - col[t][2].xc)/Δtsim
            v3 = (x3next - col[t][3].xc)/Δtsim
            v4 = (x4next - col[t][4].xc)/Δtsim
            col[t][1].vc = v1
            col[t][2].vc = v2
            col[t][3].vc = v3
            col[t][4].vc = v4
            col[t][2].ωc = col[t][3].ωc
            col[t][4].ωc = col[t][1].ωc
        end
    end
end

"""
    Fourbar doesn't work well with the generic max2mincoordinates.
"""
function max2mincoordinates_fb(maxstate)
    minstate = Vector{Float64}(undef, 4)
    q1 = UnitQuaternion(maxstate[4], maxstate[5:7])
    q2 = UnitQuaternion(maxstate[30], maxstate[31:33])  # angle 2 is body 3
    minstate[1] = Rotations.rotation_angle(q1)*sign(q1.x)*sign(q1.w)
    minstate[2] = maxstate[11]
    minstate[3] = Rotations.rotation_angle(q2)*sign(q2.x)*sign(q2.w)
    minstate[4] = maxstate[37]
    return minstate
end