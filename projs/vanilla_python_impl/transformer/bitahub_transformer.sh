cd /code
bash Mambaforge-Linux-x86_64.sh -b
source ~/mambaforge/bin/activate
mamba init
source ~/.bashrc
mamba install -c conda-forge -y cupy matplotlib
cd /code/transformer
python main.py --batch-size 128 --save-root /output/ --dataset-root ./ --num-epoch 3 --d-model 256 --num-enc-layers 3 --num-dec-layers 3 --d-ff 512 --log-every-batches 50
