include("utils.jl")


root = dirname(dirname(@__FILE__))
config = Dict{String, Vector{Float64}}()
checkpoint = loadcheckpoint("params_final")

for (id, subdict) in checkpoint    
    params = subdict["params"]
    error = subdict["kstep_mse"]
    params = params[error.!==nothing]
    error = error[error.!==nothing]
    bestparams = params[argmin(error)]
    config[id] = bestparams
end

open(joinpath(root, "config", "config.json"),"w") do f
    JSON.print(f, config)
end

println("Config saved!")
