conda create -n openmm python=3.10 -y
conda activate openmm
conda install -c conda-forge openmm=8.0 parmed ipython jupyterlab mdanalysis nglview 'ipywidgets>=7.6.0,<8' -y
# if you have a GPU, you can install the cudatoolkit
# test openmm installation
python -m openmm.testInstallation
