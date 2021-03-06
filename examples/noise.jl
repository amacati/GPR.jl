using LinearAlgebra
using GPR

include("utils/utils.jl")
include("utils/data/datasets.jl")
include("utils/data/transformations.jl")
include("utils/predictdynamics.jl")
include("parallel/core.jl")

include("maximal_coordinates/P1noise.jl")
include("maximal_coordinates/P2noise.jl")
include("maximal_coordinates/CPnoise.jl")
include("maximal_coordinates/FBnoise.jl")

include("minimal_coordinates/P1noise.jl")
include("minimal_coordinates/P2noise.jl")
include("minimal_coordinates/CPnoise.jl")
include("minimal_coordinates/FBnoise.jl")

include("baseline.jl")



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
    Σ = Dict("x" => 1e-3, "q" => 1e-3, "v" => 1e-3, "ω" => 1e-3)
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

config = Dict("nruns" => 100, "Δtsim" => 0.0001, "testsamples" => 100, "simsteps" => 20)

samplesizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]  # [2, 4, 8, 16, 32, 64, 128, 256, 512, 
dfkeys = ["P1", "P2", "CP", "FB"]
@info "Main Thread: loading datasets $dfkeys"
datasets = Dict(key => loaddatasets(key) for key in dfkeys)
@info "Main Thead: finished loading datasets"


@info "Processing pure variational integrator experiments"
parallelsim(experimentVarIntP1, expand_config("P1_MIN", 2, config, datasets["P1"]), idmod = "VI")
parallelsim(experimentVarIntP2, expand_config("P2_MIN", 2, config, datasets["P2"]), idmod = "VI")
parallelsim(experimentVarIntCP, expand_config("CP_MIN", 2, config, datasets["CP"]), idmod = "VI")
parallelsim(experimentVarIntFB, expand_config("FB_MIN", 2, config, datasets["FB"]), idmod = "VI")


for nsamples in samplesizes
    @info "Processing P1 experiments"
    parallelsim(experiment_p1_mz_max, expand_config("P1_MAX", nsamples, config, datasets["P1"]))
    parallelsim(experiment_p1_mz_min, expand_config("P1_MIN", nsamples, config, datasets["P1"]))
    parallelsim(experiment_p1_mz_min_sin, expand_config("P1_MIN", nsamples, config, datasets["P1"]), idmod = "sin")
    parallelsim(experiment_p1_md_max, expand_config("P1_MAX", nsamples, config, datasets["P1"]), idmod = "MD")
    parallelsim(experiment_p1_md_min, expand_config("P1_MIN", nsamples, config, datasets["P1"]), idmod = "MD")
    parallelsim(experiment_p1_md_min_sin, expand_config("P1_MIN", nsamples, config, datasets["P1"]), idmod = "MDsin")
end

for nsamples in samplesizes
    @info "Processing P2 experiments"
    parallelsim(experiment_p2_mz_max, expand_config("P2_MAX", nsamples, config, datasets["P2"]))
    parallelsim(experiment_p2_mz_min, expand_config("P2_MIN", nsamples, config, datasets["P2"]))
    parallelsim(experiment_p2_mz_min_sin, expand_config("P2_MIN", nsamples, config, datasets["P2"]), idmod = "sin")
    parallelsim(experiment_p2_md_max, expand_config("P2_MAX", nsamples, config, datasets["P2"]), idmod = "MD")
    parallelsim(experiment_p2_md_min, expand_config("P2_MIN", nsamples, config, datasets["P2"]), idmod = "MD")
    parallelsim(experiment_p2_md_min_sin, expand_config("P2_MIN", nsamples, config, datasets["P2"]), idmod = "MDsin")
end

for nsamples in samplesizes
    @info "Processing CP experiments"
    parallelsim(experiment_cp_mz_max, expand_config("CP_MAX", nsamples, config, datasets["CP"]))
    parallelsim(experiment_cp_mz_min, expand_config("CP_MIN", nsamples, config, datasets["CP"]))
    parallelsim(experiment_cp_mz_min_sin, expand_config("CP_MIN", nsamples, config, datasets["CP"]), idmod = "sin")
    parallelsim(experiment_cp_md_max, expand_config("CP_MAX", nsamples, config, datasets["CP"]), idmod = "MD")
    parallelsim(experiment_cp_md_min, expand_config("CP_MIN", nsamples, config, datasets["CP"]), idmod = "MD")
    parallelsim(experiment_cp_md_min_sin, expand_config("CP_MIN", nsamples, config, datasets["CP"]), idmod = "MDsin")
end

for nsamples in samplesizes
    @info "Processing FB experiments"
    parallelsim(experiment_fb_mz_max, expand_config("FB_MAX", nsamples, config, datasets["FB"]))
    parallelsim(experiment_fb_mz_min, expand_config("FB_MIN", nsamples, config, datasets["FB"]))
    parallelsim(experiment_fb_mz_min_sin, expand_config("FB_MIN", nsamples, config, datasets["FB"]), idmod = "sin")
    parallelsim(experiment_fb_md_max, expand_config("FB_MAX", nsamples, config, datasets["FB"]), idmod = "MD")
    parallelsim(experiment_fb_md_min, expand_config("FB_MIN", nsamples, config, datasets["FB"]), idmod = "MD")
    parallelsim(experiment_fb_md_min_sin, expand_config("FB_MIN", nsamples, config, datasets["FB"]), idmod = "MDsin")
end
