import subprocess
from setuptools import setup, find_packages

setup(
    name="torchmd-net",
    version="0",
    packages=find_packages(),
    entry_points={"console_scripts": ["torchmd-train = torchmdnet.scripts.train:main"]},
)
