from solver import FCIQMC, build_hamiltonian
from extrapolation_utils import inverse_fit, plot_extrapolation_to_inf_det
from pyscf import gto
import numpy as np

SHIFT = 0.7
INIT_N_WALKER = 1

if __name__ == "__main__":
    mol_2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0)
    mol_3 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g(d)', verbose=0)
    mol_4 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g(d,p)', verbose=0)
    mol_5 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-311G(d,p)', verbose=0)
    mols = [mol_2, mol_3, mol_4, mol_5]

    n_steps = 40000
    E = []
    sd = []
    N_det = np.zeros(len(mols))

    for i, mol in enumerate(mols):
        print(f"Starting FCIQMC with the {mol.basis} basis set.")
        
        # We need the number of determinants, but it could not be obtained from mol.
        # Therefore, although redundant, we construct the FCI Hamiltonian
        # and refer to the dimension of the resulting matrix.
        H_mat, _, _ = build_hamiltonian(mol)

        E_est, E_sd, E_fci, E_HF, avg_shift, hist_population, hist_shift, hist_energy = \
            FCIQMC(mol, shift=0.7, init_n_walker=3, n_steps=n_steps, n_prod=n_steps//4, step_starting_shift_upd=n_steps//4, debug=False)
        E.append(E_est)
        sd.append(E_sd)
        N_det[i] = H_mat.shape[0]

    print("All FCIQMC calculations have completed.")

    _, E_inf, fn, _, E_sd = inverse_fit(N_det, E)
    print(f"E_inf: {E_inf}")

    plot_extrapolation_to_inf_det(N_det, E, E_sd, fn)