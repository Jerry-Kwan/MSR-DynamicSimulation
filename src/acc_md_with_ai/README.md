Firstly, follow the environment setup instruction in torchmd-net

mamba install -c conda-forge -c acellera moleculekit openmm=8.0 mdtraj parmed ipython jupyterlab mdanalysis nglview 'ipywidgets>=7.6.0,<8' -y

use `pip install -e .` to install `mymd`

能量都是有参考点的，差距大是合理的

torchmd-net-main-20230515 是 20230515 download 下来的版本，删除了一些用不着的东西比如 tests/，修改了 setup.py