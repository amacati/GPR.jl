using DataFrames
using Serialization
using GPR

include("utils.jl")
include("generatedata.jl")


function generateP1dataset(config::Dict; friction::Bool=false)
    traindf = DataFrame(df = Vector{DataFrame}(), m = Vector{Float64}(), ΔJ = Vector{SMatrix}(), friction = Vector{Float64}())
    testdf = DataFrame(df = Vector{DataFrame}(), m = Vector{Float64}(), ΔJ = Vector{SMatrix}(), friction = Vector{Float64}())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"])")
        frict = friction ? rand() : 0.
        ΔJ = SMatrix{3,3,Float64}(config["Σ"]["J"]randn(9)...)
        m = abs(1. + config["Σ"]["m"]randn())
        exp1 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5) * π, ωstart = 2(rand()-0.5), m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        exp2 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=((rand()/2 + 0.5)*rand((-1,1))) * π, ωstart = 2(rand()-0.5), m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]  # [-π:-π/2; π/2:π] [-π:π]
        exptest = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5)*2π, ωstart = 2(rand()-0.5), m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        _traindf, _testdf = generate_dataframes(config, config["trainsamples"], exp1, exp2, exptest, parallel = true)
        push!(traindf, (_traindf, m, ΔJ, frict))
        push!(testdf, (_testdf, m, ΔJ, frict))
    end
    return traindf, testdf
end

function generateP2dataset(config; friction=false)
    traindf = DataFrame(df = Vector{DataFrame}(), m = Vector{Vector{Float64}}(), ΔJ = Vector{Vector{SMatrix}}(), friction = Vector{Vector{Float64}}())
    testdf = DataFrame(df = Vector{DataFrame}(), m = Vector{Vector{Float64}}(), ΔJ = Vector{Vector{SMatrix}}(), friction = Vector{Vector{Float64}}())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"])")
        frict = friction ? rand(2) : [0., 0.]
        ΔJ = [SMatrix{3,3,Float64}(config["Σ"]["J"]randn(9)...), SMatrix{3,3,Float64}(config["Σ"]["J"]randn(9)...)]
        m = abs.(ones(2) .+ config["Σ"]["m"]randn(2))
        exp1 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5) .* [π, 2π], m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        exp2 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=[(rand()/2 + 0.5)*rand([-1,1]), 2(rand()-0.5)] .* π, m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]    # [-π:-π/2; π/2:π] [-π:π]
        exptest = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5).*2π, m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        _traindf, _testdf = generate_dataframes(config, config["trainsamples"], exp1, exp2, exptest, parallel = true)
        push!(traindf, (_traindf, m, ΔJ, frict))
        push!(testdf, (_testdf, m, ΔJ, frict))
    end
    return traindf, testdf
end

function generateCPdataset(config; friction=false)
    traindf = DataFrame(df = Vector{DataFrame}(), m = Vector{Vector{Float64}}(), ΔJ = Vector{Vector{SMatrix}}(), friction = Vector{Vector{Float64}}())
    testdf = DataFrame(df = Vector{DataFrame}(), m = Vector{Vector{Float64}}(), ΔJ = Vector{Vector{SMatrix}}(), friction = Vector{Vector{Float64}}())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"])")
        ΔJ = [SMatrix{3,3,Float64}(config["Σ"]["J"]randn(9)...), SMatrix{3,3,Float64}(config["Σ"]["J"]randn(9)...)]
        m = abs.(ones(2) .+ config["Σ"]["m"]randn(2))
        frict = friction ? rand(2) .* [4., 0.3] : [0., 0.]
        nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
        exp1 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()-0.5)π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        exp2 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()/2+0.5)*rand([-1,1])π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        exptest = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=2π*(rand()-0.5), vstart=2(rand()-0.5), ωstart=2(rand()-0.5), m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        _traindf, _testdf = generate_dataframes(config, config["trainsamples"], exp1, exp2, exptest, parallel = true)
        push!(traindf, (_traindf, m, ΔJ, frict))
        push!(testdf, (_testdf, m, ΔJ, frict))
    end
    return traindf, testdf
end

function generateFBdataset(config; friction=false)
    traindf = DataFrame(df = Vector{DataFrame}(), m = Vector{Float64}(), ΔJ = Vector{SMatrix}(), friction = Vector{Vector{Float64}}())
    testdf = DataFrame(df = Vector{DataFrame}(), m = Vector{Float64}(), ΔJ = Vector{SMatrix}(), friction = Vector{Vector{Float64}}())
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"])")
        ΔJ = SMatrix{3,3,Float64}(config["Σ"]["J"]randn(9)...)
        m = abs.(1 .+ config["Σ"]["m"]randn())
        frict = friction ? rand(2) .* [4., 4.] : [0., 0.]
        exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, m = m, ΔJ = ΔJ, friction=frict, threadlock = threadlock)[1]
        _traindf, _testdf = generate_dataframes(config, config["trainsamples"], exp1, exp2, exptest, parallel = true)
        push!(traindf, (_traindf, m, ΔJ, frict))
        push!(testdf, (_testdf, m, ΔJ, frict))
    end
    return traindf, testdf
end

function savedatasets(id, traindf, testdf)
    root = joinpath(dirname(@__FILE__), "datasets")
    if !Base.Filesystem.isdir(root)
        Base.Filesystem.mkpath(root)
    end
    path = joinpath(root, id*"_trainset.jls")
    serialize(path, traindf)
    path = joinpath(root, id*"_testset.jls")
    serialize(path, testdf)
end

function loaddatasets(id)
    root = joinpath(dirname(@__FILE__), "datasets")
    path = joinpath(root, id*"_trainset.jls")
    traindf = deserialize(path)
    path = joinpath(root, id*"_testset.jls")
    testdf = deserialize(path)
    return traindf, testdf
end

function getconfig()
    Σ = Dict("m" => 1e-1, "J" => 1e-2)
    config = Dict("Δtsim" => 0.001, "Ndfs" => 100, "trainsamples" => 1000, "testsamples" => 1000, "simsteps" => 20, "Σ" => Σ)
    return config
end

function generatedataset(id::String, config::Dict; friction::Bool = false)
    occursin("P1", id) && return generateP1dataset(config, friction = friction)
    occursin("P2", id) && return generateP2dataset(config, friction = friction)
    occursin("CP", id) && return generateCPdataset(config, friction = friction)
    occursin("FB", id) && return generateFBdataset(config, friction = friction)
    throw(ArgumentError("Unsupported experiment ID: $id"))
end

function main()
    config = getconfig()
    for id in ["FB", "FBfriction"]  # "P1", "P1friction", "P2", "P2friction", "CP", "CPfriction", 
        traindf, testdf = generatedataset(id, config, friction = occursin("friction", id))
        savedatasets(id, traindf, testdf)
        @info ("Completed dataset $id")
    end
end

# main()