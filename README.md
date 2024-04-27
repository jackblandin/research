# Research

## Description
Since May 2, 2019, when this repository was created, it has been maintained exclusively by myself, Jack Blandin. It consists of various implementations of both existing and novel machine learning, reinforcement learning, and fairness algorithms, as well as supporting experiments for published works where I was a primary author and algorithm developer. These work was completed either (a) for my PhD research (2018-2024), or (b) for my own personal education.

## Setup

The following steps show how to setup the conda environment so that all dependencies are correctly installed, as well as how to run the Jupyter Notebooks in the context of the conda environment.

```sh
# Create the conda environment from the config file
conda env create -f=conda.yaml

# Activate the conda environment
conda activate research

# Create an IPython kernel which will allow you to run the Jupyter Notebook in the conda environment
python3.6 -m ipykernel install --user --name research

# Start the jupyter notebook
jupyter notebook
```

Then when you're in the Jupyter Notebook, select `Kernel > Change Kernel > research`.

# Release history

This is the release history for the algorithms available as Python packages.

Version | Date | Description
--- | --- | ---
0.2.0 | 02/12/2022 | Add RL algorithms as part of research module.
0.1.4 | 06/14/2019 | Initial stable release.

# Publications

Here is the code supporting published papers:

TMLR, 2024:
- [Experiment code](https://github.com/jackblandin/research/blob/master/experiments/fairness/group-fairness-in-rl-through-multi-objective-rewards--experiments.ipynb)
- [Figure generation code](https://github.com/jackblandin/research/blob/master/experiments/fairness/group-fairness-in-rl-through-multi-objective-rewards--heatmaps.ipynb)

FAccT, 2024:
- [Experiment code](https://github.com/jackblandin/research/blob/master/experiments/irl/Fair%20IRL%20Experiments.ipynb)
