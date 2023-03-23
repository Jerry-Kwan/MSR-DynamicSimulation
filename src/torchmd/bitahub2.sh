# used in bitahub debug mode
bash Miniconda3-latest-Linux-x86_64.sh # install under /root, and remember conda inito
source ~/.bashrc

# here, create a new conda env with python=3.9
# ...

# download pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# download other packages
conda install moleculekit -c acellera -c conda-forge
conda install ipython
pip install torchmd parmed
