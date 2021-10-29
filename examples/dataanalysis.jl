include("utils.jl")

EXPERIMENT_ID = "P2_2D_MAX_GGK"

success, checkpointdict = loadcheckpoint(EXPERIMENT_ID*"_FINAL")
!success && println("No checkpoint found. Please check the experiment ID")

onestep_msevec = checkpointdict["onestep_msevec"]
onestep_params = checkpointdict["onestep_params"]
onestep_params = onestep_params[onestep_msevec .!== nothing]
onestep_msevec = onestep_msevec[onestep_msevec .!== nothing]
display(onestep_params)
display(length(onestep_params))
bestparams = onestep_params[argmin(onestep_msevec)]

println("Best onestep MSE: $(minimum(onestep_msevec)) Best hyperparameters for $EXPERIMENT_ID are:")
display(bestparams)