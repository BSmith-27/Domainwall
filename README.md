# distributed_svpg

A distributed version of Stein Variational Policy Gradient

Do a git clone, run 

`python setup.py install`

then to run the program

`mpirun --allow-run-as-root -np 2 python run_main.py`

Dependencies: tensorflow 2.0, numpy, and mpi4py