# Research

## Description
Various implementations of existing machine learning, reinforcement learning, and fairness algorithms, as well as supporting experiments. These work was completed either (a) for my PhD research (2018-2024), or for my own personal education.

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

Version | Date | Description
--- | --- | ---
0.2.0 | 02/12/2022 | Add RL algorithms as part of research module.
0.1.4 | 06/14/2019 | Initial stable release.
