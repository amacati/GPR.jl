include("utils.jl")

EXPERIMENT_ID = "P1_2D_MIN_GGK"

success, checkpointdict = loadcheckpoint(EXPERIMENT_ID*"_FINAL")
!success && println("No checkpoint found. Please check the experiment ID")

onestep_msevec = checkpointdict["onestep_msevec"]
onestep_params = checkpointdict["onestep_params"]
bestparams = onestep_params[argmin(onestep_msevec)]

println("Best hyperparameters for $EXPERIMENT_ID are:")
display(bestparams)