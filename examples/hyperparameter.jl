include("generatedata.jl")
include("utils.jl")
include("dataset.jl")
include("maximal_coordinates/P1param.jl")
include("maximal_coordinates/P2param.jl")
include("maximal_coordinates/CPparam.jl")
include("maximal_coordinates/FBparam.jl")
include("minimal_coordinates/P1param.jl")
include("minimal_coordinates/P2param.jl")
include("minimal_coordinates/CPparam.jl")
include("minimal_coordinates/FBparam.jl")
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

function expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    EXPERIMENT_ID *= string(nsamples)
    nprocessed, kstep_mse, params = loadcheckpoint_or_defaults(_loadcheckpoint)
    config = Dict("EXPERIMENT_ID"=>EXPERIMENT_ID,
                  "nruns"=>config["nruns"],
                  "mechanism"=>mechanism,
                  "Δtsim"=>config["Δtsim"],
                  "traindf"=>traindf,
                  "testdf"=>testdf,
                  "nsamples"=>nsamples,
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
    nsteps = 2*Int(1/config["Δtsim"])  # Equivalent to 2 seconds
    threadlock = ReentrantLock()
    exp1 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5) * π, threadlock=threadlock)[1]  # Simulate 2 secs from random position, choose one sample
    exp2 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=((rand()/2 + 0.5)*rand((-1,1))) * π, threadlock=threadlock)[1]  # [-π:-π/2; π/2:π] [-π:π]
    exptest = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5)*2π, threadlock=threadlock)[1]  # Simulate 2 secs from random position, choose one sample
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = simplependulum2D(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentP1Max, config)  # Launch multithreaded search
end

function hyperparametersearchP2Max(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P2_MAX"
    nsteps = 2*Int(1/config["Δtsim"])
    threadlock = ReentrantLock()
    exp1 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5) .* [π, 2π], threadlock=threadlock)[1]
    exp2 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=[(rand()/2 + 0.5)*rand([-1,1]), 2(rand()-0.5)] .* π, threadlock=threadlock)[1]    # [-π:-π/2; π/2:π] [-π:π]
    exptest = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5).*2π, threadlock=threadlock)[1]
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = doublependulum2D(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentP2Max, config)  # Launch multithreaded search
end

function hyperparametersearchCPMax(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "CP_MAX"
    nsteps = 2*Int(1/config["Δtsim"])
    threadlock = ReentrantLock()
    exp1 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()-0.5)π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), threadlock=threadlock)[1]
    exp2 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()/2+0.5)*rand([-1,1])π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), threadlock=threadlock)[1]
    exptest = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=2π*(rand()-0.5), vstart=2(rand()-0.5), ωstart=2(rand()-0.5), threadlock=threadlock)[1]
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = cartpole(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentCPMax, config)
end

function hyperparametersearchFBMax(config, nsamples, _loadcheckpoint = false)
    EXPERIMENT_ID = "FB_MAX"
    nsteps = 2*Int(1/config["Δtsim"])
    threadlock = ReentrantLock()
    # limited to -π/2:π/2 because of crashes. Replicate: θstart = [-3.0305548753774603, -0.29607018191942747]
    exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, threadlock=threadlock)[1]
    exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, threadlock=threadlock)[1]
    exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, threadlock=threadlock)[1]
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = fourbar(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentFBMax, config)
end

function hyperparametersearchP1Min(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P1_MIN"
    nsteps = 2*Int(1/config["Δtsim"])
    threadlock = ReentrantLock()
    exp1 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5) * π, threadlock=threadlock)[1]  # Simulate 2 secs from random position, choose one sample
    exp2 = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=((rand()/2 + 0.5)*rand((-1,1))) * π, threadlock=threadlock)[1]  # [-π:-π/2; π/2:π] [-π:π]
    exptest = () -> simplependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand() - 0.5)*2π, threadlock=threadlock)[1]  # Simulate 2 secs from random position, choose one sample
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = simplependulum2D(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentP1Min, config)
end    

function hyperparametersearchP2Min(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "P2_MIN"
    nsteps = 2*Int(1/config["Δtsim"])
    threadlock = ReentrantLock()
    exp1 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5) .* [π, 2π], threadlock=threadlock)[1]
    exp2 = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=[(rand()/2 + 0.5)*rand([-1,1]), 2(rand()-0.5)] .* π, threadlock=threadlock)[1]    # [-π:-π/2; π/2:π] [-π:π]
    exptest = () -> doublependulum2D(nsteps, Δt=config["Δtsim"], θstart=(rand(2) .- 0.5).*2π, threadlock=threadlock)[1]
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = doublependulum2D(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentP2Min, config)
end

function hyperparametersearchCPMin(config, nsamples, _loadcheckpoint=false)
    EXPERIMENT_ID = "CP_MIN"
    nsteps = 2*Int(1/config["Δtsim"])
    threadlock = ReentrantLock()
    exp1 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()-0.5)π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), threadlock=threadlock)[1]
    exp2 = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=(rand()/2+0.5)*rand([-1,1])π, vstart=2(rand()-0.5), ωstart=2(rand()-0.5), threadlock=threadlock)[1]
    exptest = () -> cartpole(nsteps, Δt=config["Δtsim"], θstart=2π*(rand()-0.5), vstart=2(rand()-0.5), ωstart=2(rand()-0.5), threadlock=threadlock)[1]
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = cartpole(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentCPMin, config)
end

function hyperparametersearchFBMin(config, nsamples, _loadcheckpoint = false)
    EXPERIMENT_ID = "FB_MIN"
    nsteps = 2*Int(1/config["Δtsim"])
    threadlock = ReentrantLock()
    exp1 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, threadlock=threadlock)[1]
    exp2 = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, threadlock=threadlock)[1]
    exptest = () -> fourbar(nsteps, Δt=config["Δtsim"], θstart=(rand(2).-0.5)π, threadlock=threadlock)[1]
    traindf, testdf = generate_dataframes(config, nsamples, exp1, exp2, exptest, parallel = true)
    mechanism = fourbar(1; Δt=0.01)[2]  # Reset Δt to 0.01 in mechanism
    config = expand_config(EXPERIMENT_ID, config, mechanism, traindf, testdf, nsamples, _loadcheckpoint)
    parallelsearch(experimentFBMin, config)
end

config = Dict("nruns" => 100,
              "Δtsim" => 0.001,
              "testsamples" => 1000,
              "simsteps" => 20)

for nsamples in [2, 4, 8, 16, 32, 64, 128, 256, 512]
    hyperparametersearchP1Max(config, nsamples)
    hyperparametersearchP2Max(config, nsamples)
    hyperparametersearchCPMax(config, nsamples)
    hyperparametersearchFBMax(config, nsamples)

    hyperparametersearchP1Min(config, nsamples)    
    hyperparametersearchP2Min(config, nsamples)
    hyperparametersearchCPMin(config, nsamples)
    hyperparametersearchFBMin(config, nsamples)
end
