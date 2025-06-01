mamba create -n rec-pmf python=3.10 pip -y -c conda-forge &&
conda activate rec-pmf &&
mamba install -y -c conda-forge --file docker/requirements/conda_dev.txt && 
pip install -r docker/requirements/pip_dev.txt
