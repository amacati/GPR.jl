# Maximum coordinate GPs for variational integrators

## Structure of the repository
This repository underwent constant changes while writing a paper, so the overall structure is not as coherent as would be desirable. 
The README should give you a good idea where everything is located though, and the code itself contains further explanations.

### src
The [src](/src/) folder contains the [maximal coordinate projection](src/projections/implicitProjection.jl). It projects the predictions of a maximum coordinate GP to the constraints of a given mechanism.

Since the maximum coordinate GPs need a single input array, src also defines a [CState](src/CState.jl) that gathers the maximal coordinates of a mechanism into a single static array. In addition it defines the conversion from ConstrainedDynamic's mechanisms to CStates and vice versa.

[mDynamics.jl](src/mDynamics.jl) defines the dynamics mean function for GPs where the solution of ConstrainedDynamics' variational integration step is used as the mean function. It uses a cache because during the optimization of hyperparameters on the training data, the dynamics are solved many times for the same input data. Instead of recomputing we look up the values in the cache.

### examples
The bulk of our code is located in [examples](/examples/). The experiments themselves are defined in [maximal_coordinates](examples/maximal_coordinates/) and [minimal_coordinates](examples/minimal_coordinates/). Each experiment type has two files. `<EXPERIMENT_ID>noise.jl` measures the error in predicting the dynamics from noisy training data. `<EXPERIMENT_ID>param.jl` predicts the dynamics from perfect data and without model deviations. The logic is exemplary explained in [CPnoise.jl](examples/maximal_coordinates/CPnoise.jl) and in [CPparam.jl](examples/maximal_coordinates/CPparam.jl).
Each experiment needs a hyperparameter configuration as an initial guess that is further optimized on the training data during the experiments. These parameters are located in the [config.json](examples/config/config.json) file. For more information refer to the [config readme](examples/config/README).

We parallelized our experiments to execute multiple experiments in a reasonable amount ouf time. We queue multiple experiments and independently run them on worker processes. Each process adds its results to a shared result dictionary.
[parallel/core.jl](examples/parallel/core.jl) contains the core parallel execution loop. We pass the experiments and processing as callbacks to execute generic experiments in parallel. The routine includes mechanisms for checkpointing.
We also parallelized the training data generation since our sampling scheme requires us to simulate up to 200 trajectories per dataset. This ensures that the samples are uncorrelated but is computationally expensive. Since generating a dataset during each experiment run is prohibitively expensive we generate the data once (one dataframe per experiment run), save it, and the load the dataset for the experiments. The dataset generation is located in [dataframes.jl](examples/parallel/dataframes.jl). If you want to average your results over 100 runs that means creating 100 different datasets once. Every time the experiments are rerun each datasets is then used for exactly one trial.

[utils](examples/utils/) contains the experiment descriptions themselves and code related to the datasets and data transformations besides various utility functionalities. [utils/data](examples/utils/data/) generates the train/test datasets. In [datasets.jl](examples/utils/data/datasets.jl) the dataset creation functions are defined. If you enable the commented call to `main()` the datasets are generated. [simulations](examples/utils/data/simulations.jl) contains the experiment simulations with ConstrainedDynamics, and [transformations.jl](examples/utils/data/transformations.jl) defines the transformation from max to min coordinates as well as noise transformations for the datasets.
If you rerun the hyperparameter search and want to create a new config from the results, you need to rerun [createconfig.jl](examples/utils/createconfig.jl). Otherwise this file is irrelevant. [predictdynamics.jl](examples/utils/predictdynamics.jl) defines functions that predict the dynamics of a system with GPs. Since the prediction depends on the system type, these functions are tailored to our experiments.

In order to run the experiments you execute [noise.jl](examples/noise.jl). This will run experiments for all dataset sizes and write the final results to the results folder. 
> **Note:** The experiment settings such as noise levels and simulation step sizes are set by changing the config dict and in `expand_config()`.

If you want to rerun the hyperparameter search, you execute [hyperparameter.jl](examples/hyperparameter.jl). Don't forget to create a new config after finishing the run!

The experiments in [energy.jl](examples/energy.jl) are a simplified case of the 1 link pendulum and compare the energy conservation of the GP variational integrator with an explicit Euler integrator.
The experiments in [baseline.jl](examples/baseline.jl) were not used in the paper and are not thoroughly checked for bugs. 

### benchmark
Benchmark is outdated and not used anymore.

## Experiment types
Troughout the package we use abbreviations for our experiments as IDs. These are as follows:

- P1 (Simple pendulum)
- P2 (Double pendulum)
- CP (Cartpole)
- FB (Fourbar)

