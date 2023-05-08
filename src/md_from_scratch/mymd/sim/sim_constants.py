"""
References:
    1. https://github.com/torchmd/torchmd
"""

# https://docs.lammps.org/99/units.html
# 1 click = 48.88821 fmsec
# see my notes for details
TIME_FACTOR = 48.88821

# https://blog.sciencenet.cn/blog-3437453-1299927.html
# Boltmann const  boltz   0.001987191 Kcal / (mole - degree K)
# about 1.38064852e-23 * 1000 / 4.184 * 6.023 * 1e23 / 1e6
BOLTZMANN = 0.001987191

FS_2_NS = 1E-6
