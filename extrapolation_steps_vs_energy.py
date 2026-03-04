from solver import FCIQMC
from pyscf import gto
import matplotlib.pyplot as plt
import numpy as np

SHIFT = 0.7
INIT_N_WALKER = 1

if __name__ == "__main__":
    n_steps_arr = [5000, 10000, 20000, 30000, 40000]

    # Def. of the LiH molecule
    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        spin=0,      # singlet
        verbose=0
    )

    E = []
    sd = []

    for n_steps in n_steps_arr:
        E_est, E_sd, E_fci, E_HF, avg_shift, hist_population, hist_shift, hist_energy = \
            FCIQMC(mol, shift=SHIFT, init_n_walker=INIT_N_WALKER, n_steps=n_steps, n_prod=n_steps//4, step_starting_shift_upd=n_steps//4, debug=False)
        E.append(E_est)
        sd.append(E_sd)

    E_fci_arr = np.full_like(n_steps_arr, E_fci, dtype=float)
    plt.plot(n_steps_arr, E_fci_arr, linestyle="--", label=r"${E_{FCI}}$")
    plt.errorbar(n_steps_arr, E, yerr=sd, fmt='o', capsize=3)
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()