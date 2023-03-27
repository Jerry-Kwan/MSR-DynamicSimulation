# used in bitahub debug mode
bash Mambaforge-Linux-x86_64.sh # install under /root, and remember init
source ~/.bashrc

# here, create a new mamba env with python=3.9 and activate it
# ...

# download pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# download other packages for torchmd env
mamba install moleculekit -c acellera -c conda-forge
mamba install ipython
pip install torchmd parmed

########################################################
# packages used for torchmd-net env
mamba install -c conda-forge h5py matplotlib tqdm nnpops==0.2 pytorch_cluster==1.5.9 pytorch_geometric==2.0.3 pytorch_scatter==2.0.8 pytorch_sparse==0.6.10 pytorch-lightning==1.6.3 torchmetrics==0.8.2 rdkit

# Then follow the instructions in the TorchMD-NET repo.
# That is, cloning the repo, pip install -e . and follow the command in examples/ file folder.
