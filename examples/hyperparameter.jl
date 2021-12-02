using Random
using GPR

include("utils/utils.jl")
include("utils/predictdynamics.jl")
include("utils/data/datasets.jl")
include("utils/data/simulations.jl")
include("utils/data/transformations.jl")

include("parallel/core.jl")
include("parallel/dataframes.jl")

include("maximal_coordinates/P1param.jl")
include("maximal_coordinates/P2param.jl")
include("maximal_coordinates/CPparam.jl")
include("maximal_coordinates/FBparam.jl")
include("minimal_coordinates/P1param.jl")
include("minimal_coordinates/P2param.jl")
include("minimal_coordinates/CPparam.jl")
include("minimal_coordinates/FBparam.jl")


function expand_config(config::Dict, EXPERIMENT_ID::String, nsamples::Integer, dfs::Tuple{DataFrame, DataFrame})
    EXPERIMENT_ID *= string(nsamples)
    nprocessed, kstep_mse, params = 0, [], []
    config = Dict("EXPERIMENT_ID"=>EXPERIMENT_ID,
                  "nruns"=>config["nruns"],
                  "Δtsim"=>config["Δtsim"],
                  "datasets"=>dfs,
                  "trainsamples"=>nsamples,
                  "testsamples"=>config["testsamples"],
                  "simsteps"=>config["simsteps"], 
                  "nprocessed"=>nprocessed,
                  "kstep_mse"=>kstep_mse, 
                  "params"=>params,  # Optimal hyperparameters in sim. Vector of tested parameters in search
                  "mechanismlock"=>ReentrantLock(),
                  "paramlock"=>ReentrantLock(),
                  "checkpointlock"=>ReentrantLock(), 
                  "resultlock"=>ReentrantLock())
    return config
end

config = Dict("nruns" => 100, "Δtsim" => 0.0001, "testsamples" => 100, "simsteps" => 20)

dfkeys = ["P1", "P2", "CP", "FB"]  # "P1", "P2", "CP", "FB"
@info "Main Thread: loading datasets $dfkeys"
# datasets = Dict(key => loaddatasets(key) for key in dfkeys)
dfs = loaddatasets("P1_2048")
@info "Main Thead: finished loading datasets"

for nsamples in [2048]  # [2, 4, 8, 16, 32, 64, 128, 256, 512]
    parallelsearch(experimentP1Max, expand_config(config, "P1_MAX", nsamples, dfs))
    #parallelsearch(experimentP2Max, expand_config(config, "P2_MAX", nsamples, dfs))
    #parallelsearch(experimentCPMax, expand_config(config, "CP_MAX", nsamples, dfs))
    #parallelsearch(experimentFBMax, expand_config(config, "FB_MAX", nsamples, dfs))

    parallelsearch(experimentP1Min, expand_config(config, "P1_MIN", nsamples, dfs))
    #parallelsearch(experimentP2Min, expand_config(config, "P2_MIN", nsamples, dfs))
    #parallelsearch(experimentCPMin, expand_config(config, "CP_MIN", nsamples, dfs))
    #parallelsearch(experimentFBMin, expand_config(config, "FB_MIN", nsamples, dfs))
end
