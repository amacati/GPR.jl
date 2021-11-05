using ConstrainedDynamics
using Random
using StatsBase

include("utils.jl")

mutable struct Dataset

    storages::Vector{Storage}
    GPΔt::Float64
    randomseed::Integer
    
    function Dataset(storage::Storage, )
        new([storage], 0.01, abs(rand(Int16)))
    end

    function Dataset()
        new([], 0.01, abs(rand(Int16)))
    end

    function Dataset(storages::Vector{Storage})
        new(storages, 0.01, abs(rand(Int16)))
    end
end

function add(dataset1::Dataset, dataset2::Dataset)
    return Dataset(vcat(dataset1.storages, dataset2.storages))
end

function add(dataset::Dataset, storage::Storage)
    push!(dataset.storages, storage)
    return dataset
end

function sampledataset(dataset::Dataset, n::Integer; Δt::Float64 = 0.01, random::Bool = false, pseudorandom::Bool = false,
                       replace::Bool = false, exclude::Vector{<:Integer} = Vector{Integer}(), stepsahead::AbstractArray = [0, 1])
    idx = [i for i in 1:length(dataset.storages) if !(i in exclude)]
    data = [[getcstate(s, i) for i in 1:length(s.x[1])] for s in dataset.storages[idx]]  # Array of arrays containing the successive states of each storage
    nsets = length(data)
    nsample = div(n, nsets)
    nsamples = [nsample for _ in 1:nsets]
    nsamples[1:n%nsets] .+= 1  # Distribute remaining samples among first n%Nsets datasets
    tshift = Integer(round(dataset.GPΔt / Δt))
    @assert all([nsamples[i] <= length(data[i]) - maximum(stepsahead)*tshift for i in 1:nsets])  # Sets contain enough samples
    if random
        indices = _get_random_indices(dataset, data, nsets, nsamples, maximum(stepsahead)*tshift, pseudorandom, replace)
    else    
        indices = _get_linear_indices(data, nsets, nsamples, maximum(stepsahead)*tshift)
    end
    # [x(t+steps[0]), x(t+steps[1]), ...]
    length(stepsahead) == 1 && return reduce(vcat, [data[i][indices[i].+stepsahead[1]*tshift] for i in 1:nsets if length(indices[i])>0])
    return [reduce(vcat, [data[i][indices[i].+steps*tshift] for i in 1:nsets if length(indices[i])>0]) for steps in stepsahead]
end

function _get_random_indices(dataset, data, nsets, nsamples, maxshift, pseudorandom, replace)
    rng = pseudorandom ? MersenneTwister(dataset.randomseed) : Random.default_rng()  # Make consistent "random" draws for test sets
    return [StatsBase.sample(rng, 1:length(data[i]) - maxshift, nsamples[i], replace=replace) for i in 1:nsets]
end

function _get_linear_indices(data, nsets, nsamples, maxshift)
    indices = Vector{Vector{Integer}}(undef, nsets)
    for i in 1:nsets
        if nsamples[i] > 2
            indices[i] = Integer.(round.(collect(range(1, stop=length(data[i]) - maxshift, length=nsamples[i]))))
        elseif nsamples[i] == 1
            indices[i] = [Integer(round(length(data[i])/2))]
        else
            indices[i] = Vector{Integer}()
        end
    end
    return indices
end
Base.:+(dataset1::Dataset, dataset2::Dataset) = return add(dataset1, dataset2)

Base.:+(dataset::Dataset, storage::Storage) = return add(dataset, storage)

Base.length(dataset::Dataset) = return sum([length(s.x[1]) for s in dataset.storages])