# Submission to EUSIPCO 2023

This Repository is dedicated to the Author submission of to Eusipco 2023.


The skeleton of the code is similar to [previous CPJKU submissions](https://github.com/kkoutini/cpjku_dcase20) and the [PaSST](https://github.com/kkoutini/PaSST) repository.


# Setting up the Environment:


An installation of [conda](https://docs.conda.io/en/latest/miniconda.html) is required on your system.

This repo uses forked versions of [sacred](https://github.com/kkoutini/sacred) for configuration and logging, [pytorch-lightning](https://github.com/kkoutini/pytorch-lightning) as a convenient pytorch wrapper and [ba3l](https://github.com/kkoutini/ba3l) as an integrating tool 
between mongodb, sacred and pytorch lightning.

-----------------------

To setup the environment [Mamba](https://github.com/mamba-org/mamba) is recommended and faster than conda:


```
conda install mamba -n base -c conda-forge
```

Now you can import the environment from environment.yml. This might take several minutes.

```
mamba env create -f environment.yml
```

Alternatively, you can also import the environment using conda:

```
conda env create -f environment.yml
```

An environment named `dcase22_t1` has been created. Activate the environment:

```
conda activate dcase22_t1
```


Now install `sacred`, `ba3l` and `pl-lightening`:

```shell
# dependencies
pip install -e 'git+https://github.com/kkoutini/ba3l@v0.0.2#egg=ba3l'
pip install -e 'git+https://github.com/kkoutini/pytorch-lightning@v0.0.1#egg=pytorch-lightning'
pip install -e 'git+https://github.com/kkoutini/sacred@v0.0.1#egg=sacred' 
```

to log the files you can use pip install wandb 

sample run code:

```
CUDA_VISIBLE_DEVICES=0 python -m experiments.dcase22.t1.congater_ex with Lgate_last lambda_scheduler=0 da_lambda=2 training_method=par wandb=1
```

CUDA_VISIBLE_DEVICES: Select the GPU id

Lgate_last preconfigured to use Lsigmoid activation for the ConGater

lambda_scheduler: Sets the scheduler to increase lambda

da_lambda: sets the lambda for the model

training_method: sets the training algorithm to parallel which means domain adaptation and task learning at each epoch or "post" to train task first then domain adaptation with congater

wandb: log everything to weights and biases or not








