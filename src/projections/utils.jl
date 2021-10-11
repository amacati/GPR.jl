isactive(component::Component) = component.active




@inline function GtλTof!(mechanism, body::Body, eqc::EqualityConstraint)
    return ∂g∂ʳpos(mechanism, eqc, body.id)
end
