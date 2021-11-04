include("utils.jl")


root = dirname(@__FILE__)

config = Dict{String, Vector{Float64}}()

EXPERIMENT_IDs = Vector{String}()
for etype in ["CP_MAX", "P1_MAX", "P2_MAX"], nsamples in [2^i for i in 3:9]
    id = etype*string(nsamples)*"_FINAL"
    push!(EXPERIMENT_IDs, id)
    checkpoint_dir = loadcheckpoint(joinpath(root, "data", id))
    onestep_params = checkpoint_dir["onestep_params"]
    onestep_msevec = checkpoint_dir["onestep_msevec"]
    params = onestep_params[onestep_msevec.!==nothing]
    error = onestep_msevec[onestep_msevec.!==nothing]
    bestparams = params[argmin(error)]
    config[id] = bestparams
end

open(joinpath(root, "config", "config.json"),"w") do f
    JSON.print(f, config)
end

println("Config saved!")