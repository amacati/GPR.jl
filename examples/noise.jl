include("utils.jl")
include("generatedata.jl")
include("dataset.jl")
include("mDynamics.jl")
include("maximal_coordinates/P1noise.jl")
include("maximal_coordinates/P1dynNoise.jl")
include("maximal_coordinates/P2noise.jl")
include("maximal_coordinates/P2dynNoise.jl")
include("maximal_coordinates/CPnoise.jl")
include("maximal_coordinates/CPdynNoise.jl")
include("maximal_coordinates/FBnoise.jl")
include("maximal_coordinates/FBdynNoise.jl")

include("minimal_coordinates/P1noise.jl")
include("minimal_coordinates/P1dynNoise.jl")
include("minimal_coordinates/P2noise.jl")
include("minimal_coordinates/P2dynNoise.jl")
include("minimal_coordinates/CPnoise.jl")
include("minimal_coordinates/CPdynNoise.jl")
include("minimal_coordinates/FBnoise.jl")
include("minimal_coordinates/FBdynNoise.jl")
include("parallel.jl")


function loadcheckpoint_or_defaults(_loadcheckpoint)
    nprocessed = 0
    kstep_mse = []
    if _loadcheckpoint
        checkpoint = loadcheckpoint("noisy")["EXPERIMENT_ID"]
        nprocessed = checkpoint["nprocessed"]
        kstep_mse = checkpoint["kstep_mse"]
    end
    return nprocessed, kstep_mse
end

function expand_config(EXPERIMENT_ID, nsamples, config, _loadcheckpoint = false)
    EXPERIMENT_ID = EXPERIMENT_ID*string(nsamples)
    nprocessed, kstep_mse = loadcheckpoint_or_defaults(_loadcheckpoint)
    params = Vector{Float64}()  # declare in outer scope
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID]
    end
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    # Σ = Dict("x" => 0, "q" => 0, "v" => 0, "ω" => 0, "m" => 0, "J" => 0)
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
                  "mechanismlock" => ReentrantLock(),
                  "paramlock" => ReentrantLock(),
                  "resultlock" => ReentrantLock(),
                  "checkpointlock" => ReentrantLock())
    return deepcopy(config)
end

# "nruns" => 100, "Δtsim" => 0.001, "testsamples" => 1000, "simsteps" => 20

config = Dict("nruns" => 100,
              "Δtsim" => 0.001,
              "testsamples" => 1000,
              "simsteps" => 20)

for nsamples in [2, 4 , 8, 16, 32, 64, 128, 256, 512]
    parallelsim(experimentNoisyP1Max, expand_config("P1_MAX", nsamples, config))
    parallelsim(experimentNoisyP2Max, expand_config("P2_MAX", nsamples, config))
    parallelsim(experimentNoisyCPMax, expand_config("CP_MAX", nsamples, config))
    parallelsim(experimentNoisyFBMax, expand_config("FB_MAX", nsamples, config))
    parallelsim(experimentNoisyP1Min, expand_config("P1_MIN", nsamples, config))
    parallelsim(experimentNoisyP2Min, expand_config("P2_MIN", nsamples, config))
    parallelsim(experimentNoisyCPMin, expand_config("CP_MIN", nsamples, config))
    parallelsim(experimentNoisyFBMin, expand_config("FB_MIN", nsamples, config))

    # parallelsim(experimentMeanDynamicsNoisyP1Max, expand_config("P1_MAX", nsamples, config), md = true)
    # parallelsim(experimentMeanDynamicsNoisyP2Max, expand_config("P2_MAX", nsamples, config), md = true)
    # parallelsim(experimentMeanDynamicsNoisyCPMax, expand_config("CP_MAX", nsamples, config), md = true)
    # parallelsim(experimentMeanDynamicsNoisyFBMax, expand_config("FB_MAX", nsamples, config), md = true)
    # parallelsim(experimentMeanDynamicsNoisyP1Min, expand_config("P1_MIN", nsamples, config), md = true)
    # parallelsim(experimentMeanDynamicsNoisyP2Min, expand_config("P2_MIN", nsamples, config), md = true)
    # parallelsim(experimentMeanDynamicsNoisyCPMin, expand_config("CP_MIN", nsamples, config), md = true)
    # parallelsim(experimentMeanDynamicsNoisyFBMin, expand_config("FB_MIN", nsamples, config), md = true)
end
