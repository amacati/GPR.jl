using DataFrames
using Serialization

include("../utils.jl")
include("simulations.jl")
include("../../parallel/dataframes.jl")


function generateP1dataset(config::Dict)
    dfs = [DataFrame(df = Vector{DataFrame}(), Δm = Vector{Float64}(), ΔJ = Vector{Float64}(), friction = Vector{Float64}()) for _ in 1:4]
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"]) P1 datasets")
        friction = rand()
        ΔJ = 1 + config["Σ"]["ΔJ"]*2(rand()-0.5)
        Δm = 1 + config["Σ"]["Δm"]*2(rand()-0.5)
        exp1 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5) * π, ωstart = 2(rand()-0.5), Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        exp2 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=((rand()/2 + 0.5)*rand((-1,1))) * π, ωstart = 2(rand()-0.5), Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]  # [-π:-π/2; π/2:π] [-π:π]
        exptest = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5)*2π, ωstart = 2(rand()-0.5), Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        _dfs = generate_dataframes(config, exp1, exp2, exptest; generate_uniform = true)
        for (df, _df) in zip(dfs, _dfs)
            push!(df, (_df, Δm, ΔJ, friction))
        end
    end
    return dfs
end

function generateP2dataset(config::Dict)
    dfs = [DataFrame(df = Vector{DataFrame}(), Δm = Vector{Vector{Float64}}(), ΔJ = Vector{Vector{Float64}}(), friction = Vector{Vector{Float64}}()) for _ in 1:4]
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"]) P2 datasets")
        friction = rand(2) .* [2, 0.5]
        ΔJ = ones(2) + config["Σ"]["ΔJ"].*2(rand(2).-0.5)
        Δm = ones(2) + config["Σ"]["Δm"].*2(rand(2).-0.5)
        exp1 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5) .* [π, 2π], Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        exp2 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=[(rand()/2 + 0.5)*rand([-1,1]), 2(rand()-0.5)] .* π, Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]    # [-π:-π/2; π/2:π] [-π:π]
        exptest = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5).*2π, Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        _dfs = generate_dataframes(config, exp1, exp2, exptest; generate_uniform = true)
        for (df, _df) in zip(dfs, _dfs)
            push!(df, (_df, Δm, ΔJ, friction))
        end
    end
    return dfs
end

function generateCPdataset(config::Dict)
    dfs = [DataFrame(df = Vector{DataFrame}(), Δm = Vector{Vector{Float64}}(), ΔJ = Vector{Float64}(), friction = Vector{Vector{Float64}}()) for _ in 1:4]
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"]) CP datasets")
        ΔJ = 1 + config["Σ"]["ΔJ"].*2(rand()-0.5)  # Only change inertia of the pendulum
        Δm = ones(2) + config["Σ"]["Δm"]*2(rand(2).-0.5)
        friction = rand(2) .* [1., 0.5]
        nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
        exp1 = () -> cartpole(nsteps, Δt=config["Δtsim"], xstart=rand()-0.5, θstart=(rand()-0.5)π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        exp2 = () -> cartpole(nsteps, Δt=config["Δtsim"], xstart=rand()-0.5, θstart=(rand()/2+0.5)*rand([-1,1])π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        exptest = () -> cartpole(nsteps, Δt=config["Δtsim"], xstart=rand()-0.5, θstart=2π*(rand()-0.5), vstart=2(rand()-0.5), ωstart=2(rand()-0.5), Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        _dfs = generate_dataframes(config, exp1, exp2, exptest; generate_uniform = true)
        for (df, _df) in zip(dfs, _dfs)
            push!(df, (_df, Δm, ΔJ, friction))
        end
    end
    return dfs
end

function generateFBdataset(config::Dict)
    dfs = [DataFrame(df = Vector{DataFrame}(), Δm = Vector{Vector{Float64}}(), ΔJ = Vector{Vector{Float64}}(), friction = Vector{Vector{Float64}}()) for _ in 1:4]
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    for i in 1:config["Ndfs"]
        @info ("Working on $(i)/$(config["Ndfs"]) FB datasets")
        ΔJ = ones(4) + config["Σ"]["ΔJ"]*2(rand(4).-0.5)
        Δm = ones(4) + config["Σ"]["Δm"]*2(rand(4).-0.5)
        friction = rand(4) .* [2., 0.5, 2., 0.5]
        exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, Δm = Δm, ΔJ = ΔJ, friction=friction, threadlock = threadlock)[1]
        _dfs = generate_dataframes(config, exp1, exp2, exptest; generate_uniform = true)
        for (df, _df) in zip(dfs, _dfs)
            push!(df, (_df, Δm, ΔJ, friction))
        end
    end
    return dfs
end

function savedatasets(id, dfs, dfnames)
    root = joinpath(dirname(dirname(dirname(@__FILE__))), "datasets")
    if !Base.Filesystem.isdir(root)
        Base.Filesystem.mkpath(root)
    end
    for (df, dfname) in zip(dfs, dfnames)
        path = joinpath(root, id*"_"*dfname*".jls")
        serialize(path, df)
    end
end

function loaddatasets(id)
    root = joinpath(dirname(dirname(dirname(@__FILE__))), "datasets")
    path = joinpath(root, id*"_trainset.jls")
    traindf = deserialize(path)
    path = joinpath(root, id*"_testset.jls")
    testdf = deserialize(path)
    return traindf, testdf
end

function getconfig()
    Σ = Dict("Δm" => 0.1, "ΔJ" => 0.1)
    config = Dict("Δtsim" => 0.0001, "Ndfs" => 100, "trainsamples" => 2048, "testsamples" => 100, "simsteps" => 20, "Σ" => Σ)
    return config
end

function generatedataset(id::String, config::Dict)
    "P1" == id && return generateP1dataset(config)
    "P2" == id && return generateP2dataset(config)
    "CP" == id && return generateCPdataset(config)
    "FB" == id && return generateFBdataset(config)
    throw(ArgumentError("Unsupported experiment ID: $id"))
end

function main()
    config = getconfig()
    for id in ["P1", "P2", "CP", "FB"]  # "P1", "P2", "CP", "FB"
        dfs = generatedataset(id, config)
        savedatasets(id, dfs, ["trainset", "testset", "trainset_uniform", "testset_uniform"])
        @info ("Completed dataset $id")
    end
end

# main()