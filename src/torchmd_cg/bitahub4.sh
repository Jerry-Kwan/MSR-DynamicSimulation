# used in bitahub debug mode with pytorch 1.11 and python 3.8
bash Mambaforge-Linux-x86_64.sh # install under /root, and remember init
source ~/.bashrc

# environment2.yml is under torchmd_cg
mamba env create -f environment2.yml
mamba activate torchmd-cg

# under my_torchmd_net
pip install -e .

# other
pip install parmed

# My way to install torchmd-cg, not use `pip install torchmd-cg`.
# Modify setup.py in torchmd_cg/ to eliminate requirements.txt in torchmd_cg/,
# which means `pip install -e .` just install torchmd_cg here.
# Manually install packages in requirements.txt except schnetpack, which isn't used.
# Installing torchmd by `pip install torchmd` will install torchvision and torch,
# which doesn't work.
# So install torchmd (modified by myself) by `pip install -e .`.
# Under torchmd_cg
pip install -e .
cd torchmd_pypi20230403_modified
pip install -e .
