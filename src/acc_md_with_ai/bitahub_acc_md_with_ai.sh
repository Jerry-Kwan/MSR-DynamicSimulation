# install mamba
cd /code
bash Mambaforge-Linux-x86_64.sh -b
source ~/mambaforge/bin/activate
mamba init
source ~/.bashrc

# install torchmd-net and corresponding dependencies
cd /code/torchmd-net
mamba env create -f environment.yml
mamba activate torchmd-net
pip install -e .

# install mymd and corresponding dependencies
mamba install -c conda-forge -c acellera moleculekit openmm=8.0 mdtraj parmed ipython jupyterlab mdanalysis nglview 'ipywidgets>=7.6.0,<8' -y
cd /code
pip install -e .  # install mymd
