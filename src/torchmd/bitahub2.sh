# used in bitahub debug mode
bash Mambaforge-Linux-x86_64.sh # install under /root, and remember init
source ~/.bashrc

# here, create a new mamba env with **python=3.9** and activate it
# ...

# download pytorch
mamba install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# download other packages for torchmd env
mamba install ipython
pip install torchmd parmed
mamba install moleculekit -c acellera -c conda-forge
