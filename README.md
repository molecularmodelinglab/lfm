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

## License
All the code is available under the MIT license.

This repo includes modified code from [torch-cubic-spline-grids](https://github.com/teamtomo/torch-cubic-spline-grids) (BSD license).