# Foundation Model for Molecular Dynamics Simulation

## Environment Setup

* **fast-transformers-master-20230522.zip**:

    archive of ***fast-transformers*** used in MoLFormer

    ***setup.py*** needs to be fixed:

    ```python
    # with open("README.rst") as f:
    with open("README.md") as f:
    ```

    and then use `pip install -e .` to install

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

    use ***data_process.ipynb*** to generate, including z and pos corresponding to train, valid and test data in ***finetune_datasets_from_molformer/qm9/***