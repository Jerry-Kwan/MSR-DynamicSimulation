import setuptools

version='0'

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="torchmd",
        version=version,
        author="Acellera",
        author_email="info@acellera.com",
        description="TorchMD. Molecular dynamics with pytorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/torchmd/torchmd/",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: POSIX :: Linux",
            "License :: OSI Approved :: MIT License",
        ],
        packages=setuptools.find_packages(include=["torchmd*"], exclude=[]),
        # package_data={"torchmd": ["config.ini", "logging.ini"],},
    )
