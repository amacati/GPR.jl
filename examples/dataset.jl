using ConstrainedDynamics
using StatsBase

include("utils.jl")

mutable struct Dataset

    storages::Vector{Storage}
    GPΔt::Float64
    
    function Dataset(storage::Storage)
        new([storage], 0.01)
    end

    function Dataset()
        new([], 0.01)
    end

    function Dataset(storages::Vector{Storage})
        new(storages, 0.01)
    end
end

function add(dataset1::Dataset, dataset2::Dataset)
    return Dataset(vcat(dataset1.storages, dataset2.storages))
end

function add(dataset::Dataset, storage::Storage)
    push!(dataset.storages, storage)
    return dataset
end

function sampledataset(dataset::Dataset, n::Integer; Δt::Float64 = 0.01, random::Bool = false, replace::Bool = false, exclude::Vector{<:Integer} = Vector{Integer}())
    idx = [i for i in 1:length(dataset.storages) if !(i in exclude)]
    data = [[reduce(vcat, getstates(s, i)) for i in 1:length(s.x[1])] for s in dataset.storages[idx]]  # Array of arrays containing the successive states of each storage
    nsets = length(data)
    nsample = div(n, nsets)
    nsamples = [nsample for _ in 1:nsets]
    nsamples[1:n%nsets] .+= 1  # Distribute remaining samples among first n%Nsets datasets
    tshift = Integer(round(Δt / dataset.GPΔt))
    @assert all([nsamples[i] <= length(data[i]) - 2tshift for i in 1:nsets])  # Sets contain enough samples
    if random  # Random indices for each storage
        indices = [StatsBase.sample(1:length(data[i]) - 2tshift, nsamples[i], replace=replace) for i in 1:nsets]
    else  # Linear spaced indices for each storage
        indices = [Integer.(round.(collect(range(1, stop=length(data[i]) - 2tshift, length=nsamples[i])))) for i in 1:nsets]
    end
    return reduce(vcat, [data[i][indices[i]] for i in 1:nsets]), reduce(vcat, [data[i][indices[i].+tshift] for i in 1:nsets]), 
           reduce(vcat, [data[i][indices[i].+2tshift] for i in 1:nsets])  # x(t-1), x(t), x(t+1)
end

Base.:+(dataset1::Dataset, dataset2::Dataset) = return add(dataset1, dataset2)

Base.:+(dataset::Dataset, storage::Storage) = return add(dataset, storage)

Base.length(dataset::Dataset) = return sum([length(s.x[1]) for s in dataset.storages])