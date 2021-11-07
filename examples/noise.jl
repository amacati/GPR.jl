include("utils.jl")
include("parallel.jl")
include("generatedata.jl")
include("maximal_coordinates/2Dpendulum.jl")
include("maximal_coordinates/2DdoublePendulum.jl")
include("maximal_coordinates/cartpole.jl")
include("minimal_coordinates/2Dpendulum.jl")
include("minimal_coordinates/2DdoublePendulum.jl")
include("minimal_coordinates/cartpole.jl")

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

function expand_config(EXPERIMENT_ID, nruns, Δtsim, nsamples, ntestsets, testsamples, 
                       simsteps, _loadcheckpoint)
    nprocessed, kstep_mse = loadcheckpoint_or_defaults(_loadcheckpoint)
    params = Vector{Float64}()  # declare in outer scope
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID]
    end
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    config = Dict("EXPERIMENT_ID" => EXPERIMENT_ID,
                  "Σ" => Σ,
                  "params" => params,
                  "nruns" => nruns,
                  "Δtsim" => Δtsim,
                  "nsamples" => nsamples,
                  "ntestsets" => ntestsets,
                  "testsamples" => testsamples,
                  "simsteps" => simsteps,
                  "nprocessed" => nprocessed,
                  "testsamples" => 1000,
                  "kstep_mse" => kstep_mse,
                  "mechanismlock" => ReentrantLock(),
                  "paramlock" => ReentrantLock(),
                  "resultlock" => ReentrantLock(),
                  "checkpointlock" => ReentrantLock())
    return config
end

function parallelsimP1Max(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P1_MAX"*string(nsamples)
    config = expand_config(EXPERIMENT_ID, config["nruns"], config["Δtsim"], nsamples, config["ntestsets"],
                           config["testsamples"], config["simsteps"], _loadcheckpoint)
    parallelsim(experimentNoisyP1Max, config)
end

function parallelsimP2Max(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P2_MAX"*string(nsamples)
    config = expand_config(EXPERIMENT_ID, config["nruns"], config["Δtsim"], nsamples, config["ntestsets"],
                           config["testsamples"], config["simsteps"], _loadcheckpoint)
    parallelsim(experimentNoisyP2Max, config)
end

function parallelsimCPMax(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "CP_MAX"*string(nsamples)
    config = expand_config(EXPERIMENT_ID, config["nruns"], config["Δtsim"], nsamples, config["ntestsets"],
                           config["testsamples"], config["simsteps"], _loadcheckpoint)
    parallelsim(experimentNoisyCPMax, config)
end

function parallelsimP1Min(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P1_MIN"*string(nsamples)
    config = expand_config(EXPERIMENT_ID, config["nruns"], config["Δtsim"], nsamples, config["ntestsets"],
                           config["testsamples"], config["simsteps"], _loadcheckpoint)
    parallelsim(experimentNoisyP1Min, config)
end

function parallelsimP2Min(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P2_MIN"*string(nsamples)
    config = expand_config(EXPERIMENT_ID, config["nruns"], config["Δtsim"], nsamples, config["ntestsets"],
                           config["testsamples"], config["simsteps"], _loadcheckpoint)
    parallelsim(experimentNoisyP2Min, config)
end

function parallelsimCPMin(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "CP_MIN"*string(nsamples)
    config = expand_config(EXPERIMENT_ID, config["nruns"], config["Δtsim"], nsamples, config["ntestsets"],
                           config["testsamples"], config["simsteps"], _loadcheckpoint)
    parallelsim(experimentNoisyCPMin, config)
end

config = Dict("nruns" => 100,
              "Δtsim" => 0.001,
              "ntestsets" => 5,
              "testsamples" => 1000,
              "simsteps" => 20)
              
for nsamples in [2, 4, 8, 16, 32, 64, 128, 256, 512]
    parallelsimP1Max(config, nsamples)
    parallelsimP2Max(config, nsamples)
    parallelsimCPMax(config, nsamples)

    parallelsimP1Min(config, nsamples)
    parallelsimP2Min(config, nsamples)
    parallelsimCPMin(config, nsamples)
end