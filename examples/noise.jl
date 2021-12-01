using LinearAlgebra
using GPR

include("utils.jl")
include("generatedata.jl")
include("generate_datasets.jl")
include("mDynamics.jl")
include("predictdynamics.jl")

include("maximal_coordinates/P1noise.jl")
include("maximal_coordinates/P1dynNoise.jl")

include("maximal_coordinates/P2noise.jl")
include("maximal_coordinates/P2dynNoise.jl")

include("maximal_coordinates/CPnoise.jl")
include("maximal_coordinates/CPdynNoise.jl")

include("maximal_coordinates/FBnoise.jl")
include("maximal_coordinates/FBdynNoise.jl")

include("minimal_coordinates/P1noise.jl")
include("minimal_coordinates/P1noiseSin.jl")
include("minimal_coordinates/P1dynNoise.jl")
include("minimal_coordinates/P1dynNoiseSin.jl")

include("minimal_coordinates/P2noise.jl")
include("minimal_coordinates/P2noiseSin.jl")
include("minimal_coordinates/P2dynNoise.jl")
include("minimal_coordinates/P2dynNoiseSin.jl")

include("minimal_coordinates/CPnoise.jl")
include("minimal_coordinates/CPnoiseSin.jl")
include("minimal_coordinates/CPdynNoise.jl")
include("minimal_coordinates/CPdynNoiseSin.jl")

include("minimal_coordinates/FBnoise.jl")
include("minimal_coordinates/FBnoiseSin.jl")
include("minimal_coordinates/FBdynNoise.jl")
include("minimal_coordinates/FBdynNoiseSin.jl")

include("baseline.jl")

include("parallel.jl")


function loadcheckpoint_or_defaults(_loadcheckpoint::Bool)
    nprocessed = 0
    kstep_mse = []
    if _loadcheckpoint
        checkpoint = loadcheckpoint("noisy")["EXPERIMENT_ID"]
        nprocessed = checkpoint["nprocessed"]
        kstep_mse = checkpoint["kstep_mse"]
    end
    return nprocessed, kstep_mse
end

function expand_config(EXPERIMENT_ID::String, nsamples::Integer, config::Dict, dfs::Tuple{DataFrame, DataFrame}; _loadcheckpoint::Bool = false)
    EXPERIMENT_ID = EXPERIMENT_ID*string(nsamples)
    nprocessed, kstep_mse = loadcheckpoint_or_defaults(_loadcheckpoint)
    params = Vector{Float64}()  # declare in outer scope
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID]
    end
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2)
    # Σ = Dict("x" => 0, "q" => 0, "v" => 0, "ω" => 0)
    config = Dict("EXPERIMENT_ID" => EXPERIMENT_ID,
                  "Σ" => Σ,
                  "params" => params,
                  "nruns" => config["nruns"],
                  "Δtsim" => config["Δtsim"],
                  "nsamples" => nsamples,
                  "testsamples" => config["testsamples"],
                  "simsteps" => config["simsteps"],
                  "nprocessed" => nprocessed,
                  "kstep_mse" => kstep_mse,
                  "projectionerror" => Vector{Float64}(),
                  "mechanismlock" => ReentrantLock(),
                  "paramlock" => ReentrantLock(),
                  "resultlock" => ReentrantLock(),
                  "checkpointlock" => ReentrantLock(),
                  "datasets" => dfs)
    return config
end

function loadalldatasets(keys)
    @info "Main Thread: loading datasets $keys"
    dfcollection = Dict()
    for key in keys
        dfcollection[key] = loaddatasets(key)
    end
    @info "Main Thead: finished loading datasets"
    return dfcollection
end

# "nruns" => 100, "Δtsim" => 0.001, "testsamples" => 1000, "simsteps" => 20

config = Dict("nruns" => 100,
              "Δtsim" => 0.0001,
              "testsamples" => 100,
              "simsteps" => 20)

samplesizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
dfkeys = ["P1", "P2", "CP", "FB"]  # "P1", "P2", "CP", "FB"
datasets = loadalldatasets(dfkeys)

#=
@info "Processing pure variational integrator experiments"
parallelsim(experimentVarIntP1, expand_config("P1_MIN", 2, config, datasets["P1"]), idmod = "VI")
parallelsim(experimentVarIntP2, expand_config("P2_MIN", 2, config, datasets["P2"]), idmod = "VI")
parallelsim(experimentVarIntCP, expand_config("CP_MIN", 2, config, datasets["CP"]), idmod = "VI")
parallelsim(experimentVarIntFB, expand_config("FB_MIN", 2, config, datasets["FB"]), idmod = "VI")

for nsamples in samplesizes
    @info "Processing maxc noise experiments"
    parallelsim(experimentNoisyP1Max, expand_config("P1_MAX", nsamples, config))
    parallelsim(experimentNoisyP2Max, expand_config("P2_MAX", nsamples, config))
    parallelsim(experimentNoisyCPMax, expand_config("CP_MAX", nsamples, config))
    parallelsim(experimentNoisyFBMax, expand_config("FB_MAX", nsamples, config))
end

for nsamples in samplesizes
    @info "Processing minc noise experiments"
    parallelsim(experimentNoisyP1Min, expand_config("P1_MIN", nsamples, config, datasets["P1"]))
    parallelsim(experimentNoisyP2Min, expand_config("P2_MIN", nsamples, config, datasets["P2"]))
    parallelsim(experimentNoisyCPMin, expand_config("CP_MIN", nsamples, config, datasets["CP"]))
    parallelsim(experimentNoisyFBMin, expand_config("FB_MIN", nsamples, config, datasets["FB"]))
end

for nsamples in samplesizes
    @info "Processing minc sin noise experiments"
    parallelsim(experimentNoisyP1MinSin, expand_config("P1_MIN", nsamples, config, datasets["P1"]), idmod = "sin")
    parallelsim(experimentNoisyP2MinSin, expand_config("P2_MIN", nsamples, config, datasets["P2"]), idmod = "sin")
    parallelsim(experimentNoisyCPMinSin, expand_config("CP_MIN", nsamples, config, datasets["CP"]), idmod = "sin")
    parallelsim(experimentNoisyFBMinSin, expand_config("FB_MIN", nsamples, config, datasets["FB"]), idmod = "sin")
end

for nsamples in samplesizes
    @info "Processing maxc dynamics experiments"
    parallelsim(experimentMeanDynamicsNoisyP1Max, expand_config("P1_MAX", nsamples, config, datasets["P1"]), idmod = "MD")
    parallelsim(experimentMeanDynamicsNoisyP2Max, expand_config("P2_MAX", nsamples, config, datasets["P2"]), idmod = "MD")
    parallelsim(experimentMeanDynamicsNoisyCPMax, expand_config("CP_MAX", nsamples, config, datasets["CP"]), idmod = "MD")
    parallelsim(experimentMeanDynamicsNoisyFBMax, expand_config("FB_MAX", nsamples, config, datasets["FB"]), idmod = "MD")
end

for nsamples in samplesizes
    @info "Processing minc dynamics experiments"
    parallelsim(experimentMeanDynamicsNoisyP1Min, expand_config("P1_MIN", nsamples, config, datasets["P1"]), idmod = "MD")
    parallelsim(experimentMeanDynamicsNoisyP2Min, expand_config("P2_MIN", nsamples, config, datasets["P2"]), idmod = "MD")
    parallelsim(experimentMeanDynamicsNoisyCPMin, expand_config("CP_MIN", nsamples, config, datasets["CP"]), idmod = "MD")
    parallelsim(experimentMeanDynamicsNoisyFBMin, expand_config("FB_MIN", nsamples, config, datasets["FB"]), idmod = "MD")
end

for nsamples in samplesizes
    @info "Processing minc sin dynamics experiments"
    parallelsim(experimentMeanDynamicsNoisyP1MinSin, expand_config("P1_MIN", nsamples, config, datasets["P1"]), idmod = "MDsin")
    parallelsim(experimentMeanDynamicsNoisyP2MinSin, expand_config("P2_MIN", nsamples, config, datasets["P2"]), idmod = "MDsin")
    parallelsim(experimentMeanDynamicsNoisyCPMinSin, expand_config("CP_MIN", nsamples, config, datasets["CP"]), idmod = "MDsin")
    parallelsim(experimentMeanDynamicsNoisyFBMinSin, expand_config("FB_MIN", nsamples, config, datasets["FB"]), idmod = "MDsin")
end
=#