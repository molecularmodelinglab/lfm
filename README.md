# Ligand Force Matching (LFM)

![Overview figure](overview.svg)

This repository contains all the code for the Ligand Force Matching (LFM) workflow. See the [paper](https://arxiv.org/abs/2506.00593) for more details.

## Setting up environment
Once you have conda (or mamba, preferred) installed, simply run:
```bash
source env.sh
```
This will create the `lfm` conda environment.

## Reproducing paper metrics
The outputs from running the benchmarking virtual screening campaigns can be found [here](https://zenodo.org/records/15595314).
Once you've downloaded it, simply run:
```bash
python -m analysis.results /path/to/downloaded/folder
```

## Running the LFM workflow on a new target
The data generation + training code is pretty specific to our infrasctuture and we're in the process of cleaning it up + documenting it. It currently involves orchestrating hundreds of instances on [Vast.ai](https://vast.ai/) to do the data generation and running the training on our SLURM cluster so there are a lot of moving parts. 

In the meantime, please [reach out](mailto:mixarcid@unc.edu) if you're interesting in using this for your work! 

## License
All the code is available under the MIT license.

This repo includes modified code from [torch-cubic-spline-grids](https://github.com/teamtomo/torch-cubic-spline-grids) (BSD license).