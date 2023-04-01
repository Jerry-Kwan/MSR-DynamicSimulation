# used in bitahub debug mode with pytorch 1.11 and python 3.8
bash Mambaforge-Linux-x86_64.sh # install under /root, and remember init
source ~/.bashrc

mamba create -n torchmd_cg python=3.9
mamba activate torchmd_cg

# download pytorch
mamba install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# other
mamba install -c conda-forge -c acellera pyyaml ipython scikit-learn tqdm pytorch-lightning==1.6.3 moleculekit seaborn pandas jupyter
pip install torchmd-cg parmed
