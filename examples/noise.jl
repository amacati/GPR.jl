include("utils.jl")
include("parallelsim.jl")
include("generatedata.jl")
include("maximal_coordinates/2Dpendulum.jl")
include("maximal_coordinates/2DdoublePendulum.jl")
include("maximal_coordinates/cartpole.jl")
include("minimal_coordinates/2Dpendulum.jl")
include("minimal_coordinates/2DdoublePendulum.jl")
include("minimal_coordinates/cartpole.jl")

function get_sim_checkpoint(_loadcheckpoint)
    nprocessed = 0
    kstep_mse = []
    if _loadcheckpoint
        checkpoint = loadcheckpoint("noisy")["EXPERIMENT_ID"]
        nprocessed = checkpoint["nprocessed"]
        kstep_mse = checkpoint["kstep_mse"]
    end
    return nprocessed, kstep_mse
end

function initsim(ID, nsamples, nruns, simsteps,  _loadcheckpoint)
    # Generate UUID
    EXPERIMENT_ID = ID*string(nsamples)
    # Load checkpoints
    nprocessed, kstep_mse = get_sim_checkpoint(_loadcheckpoint)
    # Load parameters
    params = nothing
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID*"_FINAL"]
    end
    EXPERIMENT_ID *= "_NOISE"  # Change ID to avoid overwriting old results
    # Create config dictionary
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    # Σ = Dict("x" => 0, "q" => 0, "v" => 0, "ω" => 0, "m" => 0, "J" => 0)
    config = Dict("EXPERIMENT_ID" => EXPERIMENT_ID,
                  "Σ" => Σ,
                  "params" => params,
                  "nsamples" => nsamples,
                  "nprocessed" => nprocessed,
                  "nruns" => nruns,
                  "simsteps" => simsteps,
                  "testsamples" => 1000,
                  "kstep_mse" => kstep_mse,
                  "paramlock" => ReentrantLock(),
                  "resultlock" => ReentrantLock(),
                  "checkpointlock" => ReentrantLock())
    return config
end

function parallelsimP1Max(nsamples, nruns, simsteps, _loadcheckpoint=false)
    config = initsim("P1_MAX", nsamples, nruns, simsteps, _loadcheckpoint)
    # Launch parallel simulation
    parallelsim(experimentNoisyP1Max, config)
end

function parallelsimP2Max(nsamples, nruns, simsteps, _loadcheckpoint=false)
    config = initsim("P2_MAX", nsamples, nruns, simsteps, _loadcheckpoint)
    # Launch parallel simulation
    parallelsim(experimentNoisyP2Max, config)
end

function parallelsimCPMax(nsamples, nruns, simsteps, _loadcheckpoint=false)
    config = initsim("CP_MAX", nsamples, nruns, simsteps, _loadcheckpoint)
    # Launch parallel simulation
    parallelsim(experimentNoisyCPMax, config)
end

function parallelsimP1Min(nsamples, nruns, simsteps, _loadcheckpoint=false)
    config = initsim("P1_MIN", nsamples, nruns, simsteps, _loadcheckpoint)
    # Launch parallel simulation
    parallelsim(experimentNoisyP1Min, config)
end

function parallelsimP2Min(nsamples, nruns, simsteps, _loadcheckpoint=false)
    config = initsim("P2_MIN", nsamples, nruns, simsteps, _loadcheckpoint)
    # Launch parallel simulation
    parallelsim(experimentNoisyP2Min, config)
end

function parallelsimCPMin(nsamples, nruns, simsteps, _loadcheckpoint=false)
    config = initsim("P2_MIN", nsamples, nruns, simsteps, _loadcheckpoint)
    # Launch parallel simulation
    parallelsim(experimentNoisyCPMin, config)
end

nruns = 1
simsteps = 50
for nsamples in [2^i for i in 1:1]  # [2^i for i in 1:9]
    #parallelsimP1Max(nsamples, nruns, simsteps)
    # parallelsimP2Max(nsamples, nruns, simsteps)
    parallelsimCPMax(nsamples, nruns, simsteps)

    # parallelsimP1Min(nsamples, nruns)
    # parallelsimP2Min(nsamples, nruns)
    # parallelsimCPMin(nsamples, nruns)
end