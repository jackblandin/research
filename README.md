# Research

## Description
Various implementations of existing machine learning and reinforcement learning algorithms. These algorithms are meant for personal instructive purposes rather and should never be used in a production setting.

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
0.0.2 | 06/27/2019 | Use integer observations instead of one-hots.
0.0.1 | - | Initial release
