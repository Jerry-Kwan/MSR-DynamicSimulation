# used in bitahub debug mode with pytorch 1.11 and python 3.8
bash Mambaforge-Linux-x86_64.sh # install under /root, and remember init
source ~/.bashrc

# environment.yml is under my_torchmd_net
mamba env create -f environment.yml
mamba activate torchmd-net

# Then follow the instructions in the TorchMD-NET repo.
# That is, cloning the repo, pip install -e . and follow the command in examples/ file folder.
