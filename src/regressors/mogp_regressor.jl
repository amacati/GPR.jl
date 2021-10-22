struct MOGaussianProcessRegressor

    gprs::Vector{GaussianProcessRegressor}
    ngprs::Integer

    function MOGaussianProcessRegressor(gprs::Vector{GaussianProcessRegressor})
        new(gprs, length(gprs))
    end
end

Base.iterate(mogpr::MOGaussianProcessRegressor, s=1) = s > mogpr.ngprs ? nothing : (mogpr.gprs[s], s+1)

function predict(gprs::MOGaussianProcessRegressor, xstar::Vector{SVector{S,T}}) where {S,T}
    μ, σ = Vector{SVector{length(gprs),Float64}}(undef, length(xstar)), Vector{SVector{length(gprs),Float64}}(undef, length(xstar))
    μtmp, σtmp = zeros(length(xstar), length(gprs)), zeros(length(xstar), length(gprs))
    for (id, gpr) in enumerate(gprs)
        μi, σi = predict(gpr, xstar)
        μtmp[:,id] = μi
        σtmp[:,id] = σi
    end
    for id in 1:length(xstar)
        μ[id] = SVector{length(gprs), Float64}(μtmp[id,:])
        σ[id] = SVector{length(gprs), Float64}(σtmp[id,:])
    end
    return μ, σ  # σ is a vector of the diagonal elements of the covariance matrix
end

function predict(gprs::MOGaussianProcessRegressor, xstar::SVector{S, T}) where {S,T}
    return predict(gprs, [xstar,])
end

function Base.length(s::MOGaussianProcessRegressor)
    return length(s.gprs)
end

function Base.getindex(s::MOGaussianProcessRegressor, i)
    return s.gprs[i]
end

function Base.setindex!(s::MOGaussianProcessRegressor, v::GaussianProcessRegressor, i)
    s[i] = v
end

function Base.firstindex(_::MOGaussianProcessRegressor)
    return 1
end

function Base.lastindex(s::MOGaussianProcessRegressor)
    return length(s.gprs)
end