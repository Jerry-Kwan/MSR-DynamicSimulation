# TorchMD Tutorial

Reimplement paper `TorchMD: A deep learning framework for molecular simulations`

Including the following repositories:

* https://github.com/torchmd/torchmd: examples/tutorial.ipynb

    don't need to clone the code, don't need to modify code

* https://github.com/torchmd/torchmd-net: examples/

    clone 下来跑，当前代码 clone 日期为 20230326，只删了一些没用的东西，和改了一下 setup.py（不改其实也行），由于改动不大，不对当时原仓库代码进行备份

* https://github.com/torchmd/torchmd-cg: tutorial/Chignolin_Coarse-Grained_Tutorial.ipynb

    clone 下来跑，且由于环境配置原因需要修改 setup.py，并且也是由于环境配置原因需要一份 torchmd 源码用于 `pip install -e .`，并且对 torchmd 的 pyproject.toml 的 dependencies 作了修改，考虑到这个没啥必要对当时原仓库代码进行备份，就不备份了
    
    由于原来的 yaml 文件不适配当下的 torchmd-net，所以我添加了新的 yaml 文件，以及部分参考 `Chignolin_Coarse-Grained_Tutorial.ipynb` 编写的更适合在服务器与本地跑的代码文件