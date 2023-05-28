mamba create -n md-scratch python=3.9.7
mamba activate md-scratch

mamba install -c conda-forge openmm=8.0 parmed ipython jupyterlab mdanalysis nglview 'ipywidgets>=7.6.0,<8' -y
python -m openmm.testInstallation

mamba install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge
mamba install moleculekit jupyter mdtraj -c acellera -c conda-forge

pip install -e .
