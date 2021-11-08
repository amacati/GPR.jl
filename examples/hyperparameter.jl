include("generatedata.jl")
include("utils.jl")
include("dataset.jl")
include("maximal_coordinates/2Dpendulum.jl")
include("maximal_coordinates/2DdoublePendulum.jl")
include("maximal_coordinates/cartpole.jl")
include("maximal_coordinates/fourbar.jl")
include("minimal_coordinates/2Dpendulum.jl")
include("minimal_coordinates/2DdoublePendulum.jl")
include("minimal_coordinates/cartpole.jl")
include("minimal_coordinates/fourbar.jl")
include("parallel.jl")


function loadcheckpoint_or_defaults(_loadcheckpoint)
    nprocessed = 0
    kstep_mse = []
    params = []
    if _loadcheckpoint
        checkpointdict = loadcheckpoint(EXPERIMENT_ID)
        nprocessed = checkpointdict["nprocessed"]
        kstep_mse = checkpointdict["kstep_mse"]
        params = checkpointdict["params"]
    end
    return nprocessed, kstep_mse, params
end

function expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    EXPERIMENT_ID *= string(nsamples)
    nprocessed, kstep_mse, params = loadcheckpoint_or_defaults(_loadcheckpoint)
    config = Dict("EXPERIMENT_ID"=>EXPERIMENT_ID,
                  "nruns"=>config["nruns"],
                  "mechanism"=>mechanism,
                  "Δtsim"=>config["Δtsim"],
                  "dataset"=>dataset, 
                  "nsamples"=>nsamples,
                  "ntestsets"=>config["ntestsets"],
                  "testsamples"=>config["testsamples"],
                  "simsteps"=>config["simsteps"], 
                  "nprocessed"=>nprocessed,
                  "kstep_mse"=>kstep_mse, 
                  "params"=>params,  # Optimal hyperparameters in sim. Vector of tested parameters in search
                  "paramlock"=>ReentrantLock(), 
                  "checkpointlock"=>ReentrantLock(), 
                  "resultlock"=>ReentrantLock())
    return config
end
  
function hyperparametersearchP1Max(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P1_MAX"
    dataset = Dataset()  # Dataset that is shared among processes to save time
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=config["Δtsim"], θstart=θ)
        dataset += storage
    end
    mechanism = simplependulum2D(Δt=0.01, θstart=0.)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentP1Max, config)  # Launch multithreaded search
end

function hyperparametersearchP2Max(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P2_MAX"
    dataset = Dataset()  # Dataset that is shared among processes to save time
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = doublependulum2D(Δt=config["Δtsim"], θstart=[θ1, θ2])
        dataset += storage
    end
    mechanism = doublependulum2D(Δt=0.01, θstart=[0, 0])[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentP2Max, config)  # Launch multithreaded search
end

function hyperparametersearchCPMax(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "CP_MAX"
    dataset = Dataset()  # Dataset that is shared among processes to save time
    for θstart in -π:1:π, vstart in -1:1:1, ωstart in -1:1:1
        storage, _, _ = cartpole(Δt=config["Δtsim"], θstart=θstart, vstart=vstart, ωstart=ωstart)
        dataset += storage
    end
    mechanism = cartpole(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentCPMax, config)
end

function hyperparametersearchP1Min(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P1_MIN"
    dataset = Dataset()
    for θ in -π/2:0.1:π/2
        storage, _, _ = simplependulum2D(Δt=config["Δtsim"], θstart=θ)
        dataset += storage
    end
    mechanism = simplependulum2D(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentP1Min, config)
end    

function hyperparametersearchP2Min(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P2_MIN"
    dataset = Dataset()
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = doublependulum2D(Δt=config["Δtsim"], θstart=[θ1, θ2])
        dataset += storage
    end
    mechanism = doublependulum2D(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentP2Min, config)
end

function hyperparametersearchCPMin(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "CP_MIN"
    dataset = Dataset()
    for θstart in -π:1:π, vstart in -1:1:1, ωstart in -1:1:1
        storage, _, _ = cartpole(Δt=config["Δtsim"], θstart=θstart, vstart=vstart, ωstart=ωstart)
        dataset += storage
    end
    mechanism = cartpole(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentCPMin, config)
end

function hyperparametersearchFourbarMax(config, nsamples, _loadcheckpoint = false)
    EXPERIMENT_ID = "FB_MAX"
    dataset = Dataset()
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = fourbar(Δt=config["Δtsim"], θstart=[θ1, θ2])
        dataset += storage
    end
    mechanism = fourbar(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentFourbarMax, config)
end

function hyperparametersearchFourbarMin(config, nsamples, _loadcheckpoint = false)
    EXPERIMENT_ID = "FB_MIN"
    dataset = Dataset()
    for θ1 in -π/3:0.5:π/3, θ2 in -π/3:0.5:π/3
        storage, _, _ = fourbar(Δt=config["Δtsim"], θstart=[θ1, θ2])
        dataset += storage
    end
    mechanism = fourbar(Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, dataset, nsamples, _loadcheckpoint)
    parallelsearch(experimentFourbarMin, config)
end

config = Dict("nruns" => 1,
              "Δtsim" => 0.001,
              "ntestsets" => 5,
              "testsamples" => 5,
              "simsteps" => 1)

for nsamples in [2] # , 4, 8, 16, 32, 64, 128, 256, 512]
    # hyperparametersearchP1Max(config, nsamples)
    # hyperparametersearchP2Max(config, nsamples)
    # hyperparametersearchCPMax(config, nsamples)
    # hyperparametersearchP1Min(config, nsamples)
    # hyperparametersearchP2Min(config, nsamples)
    # hyperparametersearchCPMin(config, nsamples)
    hyperparametersearchFourbarMax(config, nsamples)
    hyperparametersearchFourbarMin(config, nsamples)
end
