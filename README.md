# Full Configuration Interaction Quantum Monte Carlo (FCIQMC)

In this repository, a Python implementation of the Full Configuration Interaction Quantum Monte Carlo (FCIQMC) method, originally proposed by Booth *et al.* in [*Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space*](https://pubs.aip.org/aip/jcp/article-abstract/131/5/054106/902018/Fermion-Monte-Carlo-without-fixed-nodes-A-game-of?redirectedFrom=fulltext) is provided. This code was developed as part of an undergraduate thesis at Internatinal Christian University.

## Usage

```
$ pip install pyscf matplotlib tqdm
``` 

Perform a FCIQMC simulation.

```
$ python main.py
```

![Plot of the projected energy](figures/main_1.png)
![Plot of the shift value](figures/main_2.png)
![Plot of the walker population](figures/main_3.png)

Perform a extrapolation of the energy with respect to the number of steps.

```
$ python extrapolation_steps_vs_energy.py
```

![Plot of the extrapolation of the energy with respect to the number of steps](figures/extrapolation_steps_vs_energy.png)


Perform a extrapolation of the energy with respect to the number of determinants.

```
$ python extrapolation_det_vs_energy.py
```

![Plot of the extrapolation of the energy with respect to the number of determinants](figures/extrapolation_det_vs_energy.png)
