from solver import FCIQMC, plot_result
from pyscf import gto

SHIFT = 0.7
INIT_N_WALKER = 3
N_STEPS = 20000
N_PROD = 10000
STEP_STARTING_SHIFT_UPD = 10000

if __name__ == "__main__":
    # Definition of the H2 molecule (0.74 Å)
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)

    E_est, E_sd, E_fci, E_HF, avg_shift, hist_population, hist_shift, hist_energy = \
        FCIQMC(mol, shift=SHIFT, init_n_walker=INIT_N_WALKER, n_steps=N_STEPS, n_prod=N_PROD, step_starting_shift_upd=STEP_STARTING_SHIFT_UPD)

    print(f"E_fci={E_fci}, E_HF={E_HF}")

    # Output the result
    print("-" * 30)
    print(f"FCI Energy: {E_fci:.6f} Ha")
    print(f"FCIQMC Energy: {E_est:.6f} ± {E_sd: .6f} Ha")
    print(f"Average shift: {avg_shift:.6f} Ha => Energy: {avg_shift + E_HF}")
    plot_result(E_fci, E_HF, hist_population, hist_shift, hist_energy)
