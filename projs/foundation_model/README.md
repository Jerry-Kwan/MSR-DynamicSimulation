# Foundation Model for Molecular Dynamics Simulation

## Environment Setup

```bash
bash Mambaforge-Linux-x86_64.sh
source ~/.bashrc

mamba create --name MolTran_CUDA11 python=3.8.10
mamba activate MolTran_CUDA11

mamba install pytorch==1.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install -c conda-forge cudatoolkit-dev  # should be 11.7, used to install nvcc for apex

mamba install rdkit==2021.03.2 pandas=1.2.4 scikit-learn=0.24.2 scipy=1.6.3 

pip cache purge
pip install -i https://mirrors.aliyun.com/pypi/simple transformers==4.6.0 pytorch-lightning==1.1.5 datasets==1.6.2 jupyterlab==3.4.0 ipywidgets==7.7.0 bertviz==1.4.0

git clone git@github.com:li-car-fei/fast-transformers.git
# here you need to modify setup.py mentioned in the following section
pip install -e . --user --no-cache-dir

git clone git@github.com:NVIDIA/apex.git
cd apex
export CUDA_HOME=$CONDA_PREFIX
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# for torchmd-net
mamba install matplotlib
mamba install pyg=2.2.0 -c pyg
mamba install torchmetrics=0.8.2
mamba install -c pyg pytorch-cluster=1.6.0
mamba install -c pyg pytorch-scatter=2.0.9

# torchmd-net
pip install -e .
```

* **fast-transformers-master-20230522.zip**:

    archive of fast-transformers used in MoLFormer

    *setup.py* needs to be fixed:

    ```python
    # with open("README.rst") as f:
    with open("README.md") as f:
    ```

    and then use `pip install -e .` to install
    
* **torchmd-net-main-20230522**:

    archive of torchmd-net used in my MoLFormer

    some codes are modified due to the usage of low version of pytorch-lightning, marked with ***jk modified - env***
    
    *torchmd_t2.py* is added as a new model to implement attention between SMILES embeddings and z (atomic number) embeddings
    
    some codes are modified to merge the new model, marked with ***jk modified - new***
    
    use `pip install -e .` to install torchmd-net

## Data

* **finetune_datasets_from_molformer**:

    download from [IBM molformer](https://github.com/IBM/molformer)

* **Pretrained-MoLFormer**:

    download from [IBM molformer](https://github.com/IBM/molformer)

    too large to be uploaded to GitHub, so you need to download it from the above link

* **dsgdb9nsd.xyz**:

    download from [figshare](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)

    see [README](https://figshare.com/articles/dataset/Readme_file_Data_description_for_Quantum_chemistry_structures_and_properties_of_134_kilo_molecules_/1057641?backTo=/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) for data description for this dataset

    [some info](http://quantum-machine.org/datasets/)

    too large to be uploaded to GitHub, so you need to download it from the above link

* **qm9_z_and_pos**:

    use *data_process.ipynb* to generate, including z and pos corresponding to train, valid and test data in *finetune_datasets_from_molformer/qm9/*

## How to Run

`molformer-main-20230522/finrtune/run_myft_all_1.sh` (actually same as part_1)

`molformer-main-20230522/finrtune/run_myft_all_2.sh`

`molformer-main-20230522/finrtune/run_myft_part_1.sh`

`molformer-main-20230522/finrtune/run_myft_part_2.sh`

## Results

`data/rslt/`

* all_2: all parameters tuned + new model
* part_1: MoLFormer frozen + old model (i.e. have nothing to do with MoLFormer)
* part_2: MoLFormer frozen + new model

## References

[IBM MoLFormer](https://github.com/IBM/molformer)

[TorchMD-NET](https://github.com/torchmd/torchmd-net)

[li-car-fei fast-transformers](https://github.com/li-car-fei/fast-transformers)

Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014): Quantum chemistry structures and properties of 134 kilo molecules. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.978904.v5