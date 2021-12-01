function predictdynamics(mechanism::Mechanism, gps::Vector{<:GPE}, startobservation::CState, steps::Integer, getvω::Function; regularizer::Real = 0.)
    projectionerror = 0
    oldstates = startobservation
    setstates!(mechanism, oldstates)
    for _ in 1:steps
        obs = reshape(oldstates.state, :, 1)
        μ = [predict_y(gp, obs)[1][1] for gp in gps]
        vcurr, ωcurr = getvω(μ)
        vconst, ωconst = projectv!(vcurr, ωcurr, mechanism, regularizer=regularizer)
        projectionerror += norm(vcat(reduce(vcat, vconst .- vcurr), reduce(vcat, ωconst .-ωcurr)))
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xcurr, vcurr
        oldstates = CState(mechanism)
    end
    foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)  # Now at xnew, undef
    return CState(mechanism), projectionerror/steps
end

function predictdynamics(mechanism::Mechanism, startobservation::CState, steps::Int)
    setstates!(mechanism, startobservation)
    ConstrainedDynamics.simulate!(mechanism, 1:steps+1, Storage{Float64}())
    return CState(mechanism)
end

function predictdynamicsmin(mechanism::Mechanism, etype::String, gps::Vector{<:GPE}, startobservation::Vector{Float64}, steps::Integer; usesin::Bool = false)
    etype == "P1" && return _predictdynamicsp1min(mechanism, gps, startobservation, steps, usesin)
    etype == "P2" && return _predictdynamicsp2min(mechanism, gps, startobservation, steps, usesin)
    etype == "CP" && return _predictdynamicscpmin(mechanism, gps, startobservation, steps, usesin)
    etype == "FB" && return _predictdynamicsfbmin(mechanism, gps, startobservation, steps, usesin)
    throw(ArgumentError("Experiment $etype not supported!"))
end

function _predictdynamicsp1min(mechanism, gps, startobservation, steps, usesin)
    l = mechanism.bodies[1].shape.rh[2]
    θold, ωold = startobservation
    θcurr = θold + mechanism.Δt*ωold
    for _ in 1:steps
        obs = usesin ? reshape([sin(θold), ωold], :, 1) : reshape([θold, ωold], :, 1)
        ωcurr = predict_y(gps[1], obs)[1][1]
        θold, ωold = θcurr, ωcurr
        θcurr = θcurr + ωcurr*mechanism.Δt
    end
    q = UnitQuaternion(RotX(θcurr))
    return CState([0, 0.5l*sin(θcurr), -0.5l*cos(θcurr), q2vec(q)..., zeros(6)...])
end

function _predictdynamicsp2min(mechanism, gps, startobservation, steps, usesin)
    l1, l2 = mechanism.bodies[1].shape.xyz[3], mechanism.bodies[2].shape.xyz[3]
    θ1old, ω1old, θ2old, ω2old = startobservation
    θ1curr, θ2curr = θ1old + mechanism.Δt*ω1old, θ2old + mechanism.Δt*ω2old
    for _ in 1:steps
        obs = usesin ? reshape([sin(θ1old), ω1old, sin(θ2old), ω2old], :, 1) : reshape([θ1old, ω1old, θ2old, ω2old], :, 1)
        ω1curr, ω2curr = [predict_y(gp, obs)[1][1] for gp in gps]
        θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
        θ1curr = θ1curr + ω1curr*mechanism.Δt
        θ2curr = θ2curr + ω2curr*mechanism.Δt
    end
    q1, q2 = UnitQuaternion(RotX(θ1curr)), UnitQuaternion(RotX(θ1curr + θ2curr))
    x1 = [0, 0.5l1*sin(θ1curr), -0.5l1*cos(θ1curr)]
    x2 = [0, l1*sin(θ1curr) + 0.5l2*sin(θ1curr+θ2curr), -l1*cos(θ1curr) - 0.5l2*cos(θ1curr + θ2curr)]
    return CState([x1..., q2vec(q1)..., zeros(6)..., x2..., q2vec(q2)..., zeros(6)...])
end

function _predictdynamicscpmin(mechanism, gps, startobservation, steps, usesin)
    l = mechanism.bodies[2].shape.rh[2]
    xold, vold, θold, ωold = startobservation
    xcurr, θcurr = xold + vold*mechanism.Δt, θold + ωold*mechanism.Δt
    for _ in 1:steps
        obs = usesin ? reshape([xold, vold, sin(θold), ωold], :, 1) : reshape([xold, vold, θold, ωold], :, 1)
        vcurr, ωcurr = [predict_y(gp, obs)[1][1] for gp in gps]
        xold, vold, θold, ωold = xcurr, vcurr, θcurr, ωcurr
        xcurr = xcurr + vcurr*mechanism.Δt
        θcurr = θcurr + ωcurr*mechanism.Δt
    end
    q = UnitQuaternion(RotX(θcurr))
    return CState([0, xcurr, 0, 1, zeros(10)..., 0.5l*sin(θcurr)+xcurr, -0.5l*cos(θcurr), q2vec(q)..., zeros(6)...])
end

function _predictdynamicsfbmin(mechanism, gps, startobservation, steps, usesin)
    l = mechanism.bodies[1].shape.xyz[3]
    θ1old, ω1old, θ2old, ω2old = startobservation
    θ1curr, θ2curr = θ1old + ω1old*mechanism.Δt, θ2old + ω2old*mechanism.Δt
    for _ in 1:steps
        obs = usesin ? reshape([sin(θ1old), ω1old, sin(θ2old), ω2old], :, 1) : reshape([θ1old, ω1old, θ2old, ω2old], :, 1)
        ω1curr, ω2curr = [predict_y(gp, obs)[1][1] for gp in gps]
        θ1old, ω1old, θ2old, ω2old = θ1curr, ω1curr, θ2curr, ω2curr
        θ1curr = θ1curr + ω1curr*mechanism.Δt
        θ2curr = θ2curr + ω2curr*mechanism.Δt
    end
    x1 = [0, .5sin(θ1curr)l, -.5cos(θ1curr)l]
    x2 = [0, sin(θ1curr)l + .5sin(θ2curr)l, -cos(θ1curr)l - .5cos(θ2curr)l]
    x3 = [0, .5sin(θ2curr)l, -.5cos(θ2curr)l]
    x4 = [0, sin(θ2curr)l + 0.5sin(θ1curr)l, -cos(θ2curr)l - .5cos(θ1curr)l]
    q1 = UnitQuaternion(RotX(θ1curr))
    q2 = UnitQuaternion(RotX(θ2curr))
    return CState([x1..., q2vec(q1)..., zeros(6)..., x2..., q2vec(q2)..., zeros(6)..., x3..., q2vec(q2)..., zeros(6)..., x4..., q2vec(q1)..., zeros(6)...])
end
