# used in bitahub debug mode with pytorch 1.11 and python 3.8
bash Miniconda3-latest-Linux-x86_64.sh # install under /root, and remember conda init
source ~/.bashrc
conda install -c conda-forge openmm # cudatoolkit will also be installed

# https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin
conda install -c conda-forge gcc=12.1.0
