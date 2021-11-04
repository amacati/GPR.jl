include("utils.jl")
include("parallelsim.jl")
include("generatedata.jl")
include("maximal_coordinates/2Dpendulum.jl")
include("maximal_coordinates/2DdoublePendulum.jl")
include("maximal_coordinates/cartpole.jl")
include("minimal_coordinates/2Dpendulum.jl")
include("minimal_coordinates/2DdoublePendulum.jl")
include("minimal_coordinates/cartpole.jl")

function get_checkpoint_params(_loadcheckpoint)
    nprocessed = 0
    onestep_msevec = []
    if _loadcheckpoint
        checkpointdict = loadcheckpoint(EXPERIMENT_ID)
        nprocessed = checkpointdict["nprocessed"]
        onestep_msevec = checkpointdict["onestep_msevec"]
    end
    return nprocessed, onestep_msevec
end

function get_config(EXPERIMENT_ID, Σ, params, nsamples, nprocessed, nruns, onestep_msevec)
    config = Dict("EXPERIMENT_ID" => EXPERIMENT_ID,
                      "Σ" => Σ,
                      "params" => params,
                      "nsamples" => nsamples,
                      "nprocessed" => nprocessed,
                      "nruns" => nruns,
                      "onestep_msevec" => onestep_msevec,
                      "paramlock" => ReentrantLock(),
                      "resultlock" => ReentrantLock(),
                      "checkpointlock" => ReentrantLock())
    return config
end

function parallelsimP1Max(nsamples, nruns, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P1_MAX"*string(nsamples)

    # Load checkpoints
    nprocessed, onestep_msevec = get_checkpoint_params(_loadcheckpoint)

    # Load parameters
    params = nothing
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID*"_FINAL"]
    end
    EXPERIMENT_ID *= "_NOISE"  # Change ID to avoid overwriting old results
    # Create config dictionary
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    config = get_config(EXPERIMENT_ID, Σ, params, nsamples, nprocessed, nruns, onestep_msevec)

    # Launch parallel simulation
    parallelsim(experimentNoisyP1Max, config)
end

function parallelsimP2Max(nsamples, nruns, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P2_MAX"*string(nsamples)

    # Load checkpoints
    nprocessed, onestep_msevec = get_checkpoint_params(_loadcheckpoint)

    # Load parameters
    params = nothing
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID*"_FINAL"]
    end
    EXPERIMENT_ID *= "_NOISE"  # Change ID to avoid overwriting old results
    # Create config dictionary
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    config = get_config(EXPERIMENT_ID, Σ, params, nsamples, nprocessed, nruns, onestep_msevec)

    # Launch parallel simulation
    parallelsim(experimentNoisyP2Max, config)
end

function parallelsimCPMax(nsamples, nruns, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "CP_MAX"*string(nsamples)

    # Load checkpoints
    nprocessed, onestep_msevec = get_checkpoint_params(_loadcheckpoint)

    # Load parameters
    params = nothing
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID*"_FINAL"]
    end
    EXPERIMENT_ID *= "_NOISE"  # Change ID to avoid overwriting old results
    # Create config dictionary
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    config = get_config(EXPERIMENT_ID, Σ, params, nsamples, nprocessed, nruns, onestep_msevec)

    # Launch parallel simulation
    parallelsim(experimentNoisyCPMax, config)
end

function parallelsimP1Min(nsamples, nruns, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P1_MIN"*string(nsamples)

    # Load checkpoints
    nprocessed, onestep_msevec = get_checkpoint_params(_loadcheckpoint)

    # Load parameters
    params = nothing
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID*"_FINAL"]
    end
    EXPERIMENT_ID *= "_NOISE"  # Change ID to avoid overwriting old results
    # Create config dictionary
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    config = get_config(EXPERIMENT_ID, Σ, params, nsamples, nprocessed, nruns, onestep_msevec)

    # Launch parallel simulation
    parallelsim(experimentNoisyP1Min, config)
end

function parallelsimP2Min(nsamples, nruns, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "P2_MIN"*string(nsamples)

    # Load checkpoints
    nprocessed, onestep_msevec = get_checkpoint_params(_loadcheckpoint)

    # Load parameters
    params = nothing
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID*"_FINAL"]
    end
    EXPERIMENT_ID *= "_NOISE"  # Change ID to avoid overwriting old results
    # Create config dictionary
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    config = get_config(EXPERIMENT_ID, Σ, params, nsamples, nprocessed, nruns, onestep_msevec)

    # Launch parallel simulation
    parallelsim(experimentNoisyP2Min, config)
end

function parallelsimCPMin(nsamples, nruns, _loadcheckpoint=false)
    # Generate UUID
    EXPERIMENT_ID = "CP_MIN"*string(nsamples)

    # Load checkpoints
    nprocessed, onestep_msevec = get_checkpoint_params(_loadcheckpoint)

    # Load parameters
    params = nothing
    open(joinpath(dirname(@__FILE__), "config", "config.json"),"r") do f
        params = JSON.parse(f)[EXPERIMENT_ID*"_FINAL"]
    end
    EXPERIMENT_ID *= "_NOISE"  # Change ID to avoid overwriting old results
    # Create config dictionary
    Σ = Dict("x" => 5e-3, "q" => 5e-2, "v" => 5e-2, "ω" => 5e-2, "m" => 1e-1, "J" => 1e-2)
    config = get_config(EXPERIMENT_ID, Σ, params, nsamples, nprocessed, nruns, onestep_msevec)

    # Launch parallel simulation
    parallelsim(experimentNoisyCPMin, config)
end

nruns = 100
for nsamples in [2^i for i in 1:9]
    parallelsim1PMax(nsamples, nruns)
    parallelsim2PMax(nsamples, nruns)
    parallelsimCPMax(nsamples, nruns)

    parallelsim1PMin(nsamples, nruns)
    parallelsim2PMin(nsamples, nruns)
    parallelsimCPMin(nsamples, nruns)
end