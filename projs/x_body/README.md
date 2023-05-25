# N-Body

A simulation of a dynamical system with any number of particles interacting with each other gravitationally, using different numerical integration techniques.

## Numerical Integration Methods

* Euler
* Modified Euler
* Leapfrog (same as Modified Euler)
* scipy.integrate.odeint

## How to Run

Several examples are given here, see `n_body/main_anim.py` for more command line arguments.

Output (.gif) is in the `data` folder.

```bash
# example: run with default arguments
cd n_body
python main_anim.py

# example: run with 6 particles
cd n_body
python main_anim.py --num-particles 6

# example: run with leapfrog integrator
cd n_body
python main_anim.py --integrator leapfrog
```

## References

[知乎](https://zhuanlan.zhihu.com/p/76111855)

[Create Your Own N-body Simulation (With Python)](https://medium.com/swlh/create-your-own-n-body-simulation-with-python-f417234885e9)