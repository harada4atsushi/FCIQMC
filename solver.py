import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyscf import fci, ao2mo

rng = np.random.default_rng()

def get_dprinter(debug=True):
    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)     
    return dprint

    
def build_hamiltonian(mol):
    E_nuc = mol.energy_nuc()
    elec_num = sum(mol.nelec)
    myhf = mol.HF()
    myhf.kernel()
    mymp2 = myhf.MP2()
    mymp2.kernel() # run mp2
    C = myhf.mo_coeff # molecular coefficient matrix
    h1 = np.matmul((C.T),np.matmul((myhf.get_hcore()),(C)))
    h2 = ao2mo.kernel(mol,C)
    h2 = ao2mo.restore(8,h2,mol.nao_nr())

    H_mat = fci.direct_spin1.pspace(h1, h2, mol.nao_nr(), elec_num, np=70000)[1] # creates Hamiltonian
    H_mat = H_mat + E_nuc * np.eye(H_mat.shape[0])
    # nspatorbs = int(mol.nao_nr())

    # --- Full FCI energy ---
    cis = fci.FCI(myhf)
    E_fci, _civec = cis.kernel(h1, h2, C.shape[1], mol.nelec)
    E_HF = myhf.e_tot

    return H_mat, E_fci, E_HF


def FCIQMC(mol, shift=0, init_n_walker=1, n_steps=5000, step_starting_shift_upd=500, n_prod=2500,
           shift_upd_interval=10, dt=0.001, damping=0.05, debug=True):
    """
    Estimate the ground-state energy using Full Configuration Interaction Quantum Monte Carlo (FCIQMC) 

    Args:
        mol: Molecule object of PySCF
        shift: Initial shift value. Shift controls the walker population
        init_n_walker: Initial number of walkers on the HF determinant
        n_steps: Number of steps for the simulation
        step_starting_shift_upd: Number of steps at which the shift update starts. Shift is fixed until this steps.
        shift_upd_interval: Interval between shift udpate. If set to 10, the shift is updated every 10 steps.
        n_prod: Number of steps used to compute the average of E_proj (counted backward from the end of the run)        
        dt: Imaginary time step. Smaller values improve stability but increase computational cost.
        damping: Parameter controlling the magnitude of the shift update
        debug: debug mode
        

    Returns:
        tuple:
            - **E_est (float)**: Average of the projected energy
            - **E_sd (float)**: Standard deviation of the E_est
            - **E_fci (float)**: FCI energy 
            - **E_HF (float)**: Hartree-Fock energy
            - **avg_shift (float)**: Average of the shift value
            - **hist_population (list[int])**: History of the walker population 
            - **hist_shift (list[float])**: History of the shift value
            - **hist_energy (list[float])**: Histroy of the projected energy

    """
    dprint = get_dprinter(debug)
    
    H_mat, E_fci, E_HF = build_hamiltonian(mol)
    H_mat = H_mat - np.eye(H_mat.shape[0]) * E_HF
    dprint(f"--- K Matrix shape:{H_mat.shape} ---")
    dprint(H_mat)
    
    E_est = 0
    hist_population = np.zeros(n_steps)
    hist_shift = np.zeros(n_steps)
    hist_energy = np.zeros(n_steps)
    
    diag_H = np.diag(H_mat)    
    n_det = H_mat.shape[0]
    pre_shift = shift
    pre_pop = init_n_walker

    # Index of Hartree-Fock determinant
    # Normally, it corresponds to the first matrix element in the FCI of the pySCF
    ref_det_idx = 0 
        
    # Initialize walkers
    # Signed interger (+1, -1)
    walkers = np.zeros(n_det, dtype=np.int32)
    walkers[ref_det_idx] = init_n_walker

    tqdm_pbar = tqdm(range(n_steps))
    for step in tqdm_pbar:
        target_det_idx = np.flatnonzero(walkers)
        target_walkers = walkers[target_det_idx]
        n_target_det = target_det_idx.shape[0]
        
        # ----- Spawning Step -----
        # Randomly choose the target determinant for spawning
        # Generate target indices while avoiding the parent determinant index
        t_idx = np.random.randint(0, n_det-1, size=n_target_det)
        t_idx += (t_idx >= target_det_idx)

        # Compute spawning probabilities
        h_ij = H_mat[target_det_idx, t_idx]
        p_gen = 1/(n_det-1)
        prob_spawn = dt * abs(h_ij) / p_gen
        frac_prob_spawn, int_prob_spawn = np.modf(prob_spawn)

        # The integer part of the spawning probability leads to deterministic spawning
        n_spawn = int_prob_spawn.astype(np.int32)
        # The fractional part leads to stochastic spawning
        n_spawn += np.random.binomial(abs(target_walkers), frac_prob_spawn)

        # If h_ij > 0, spawn walkers with the opposite sign of the parent
        # If h_ij < 0, spawn walkers with the same sign as the parent
        signs = np.sign(target_walkers)
        signs[h_ij > 0] *= -1
        n_spawn *= signs

        # ----- Diagonal Step (Death/Cloning) -----
        prob_death = dt * (diag_H[target_det_idx] - shift)
        signs = np.sign(target_walkers)

        # Perform death or cloning with probability prob_death (Count the number of walkers affected)
        n_dc = np.random.binomial(abs(target_walkers), abs(prob_death))
        # If prob_death > 0, walkers are removed (death)
        # If prob_death < 0, walkers are duplicated (cloning)
        n_dc[prob_death > 0] *= -1
        target_walkers = signs * (abs(target_walkers) + n_dc)
        walkers[target_det_idx] = target_walkers

        # ----- Annihilation Step -----
        # walkers += new_walkers
        np.add.at(walkers, t_idx, n_spawn)

        # # ---- Shift Update -----
        current_pop = np.sum(abs(walkers))
        if step > step_starting_shift_upd: # After the initial equilibration phase
            if step % shift_upd_interval == 0:
                shift = pre_shift - (damping / (shift_upd_interval*dt)) * np.log(current_pop / pre_pop)
                pre_shift = shift

        pre_pop = current_pop
        hist_shift[step] = shift
        hist_population[step] = current_pop

        # ----- Energy Estimation (Projected Energy) -----
        row0 = H_mat[ref_det_idx]
        num_ref = walkers[ref_det_idx]

        # In the original paper, the sum is restricted to single and double excitations.
        # In this implementation, all elements are summed. However, in the FCI Hamiltonian
        # the matrix elements beyond singles/doubles are already zero, so the result is equivalent.
        s = row0 * walkers / num_ref
        E_proj = E_HF + np.sum(s[1:])

        hist_energy[step] = E_proj

        if step % 1000 == 0:
            tqdm_pbar.set_postfix(n_walkwers=current_pop, shift=f"{shift:.5f}", E_proj=f"{E_proj:.6f}")
            dprint(f'{step} step')
            log_n_walkers(walkers, debug=debug)

    # Divide the last n_prod projected energy samples into 10 blocks
    # and compute the mean and standard deviation from block averages
    n_block = 10
    hist_energy_prod = np.array(hist_energy[-n_prod:])
    block_size = len(hist_energy_prod) // n_block
    hist_energy_prod = hist_energy_prod[:n_block * block_size]  # Discard the remainder
    blocks = hist_energy_prod.reshape(n_block, block_size)
    block_means = blocks.mean(axis=1)
    E_sd = np.std(block_means, ddof=1)
    E_est = np.mean(block_means)
    
    avg_shift = np.mean(hist_shift[-n_prod:]) # Average shift over the last n_prod steps
    
    dprint('\n---- Walkers (result) ----')
    log_n_walkers(walkers, debug=debug)

    return E_est, E_sd, E_fci, E_HF, avg_shift, hist_population, hist_shift, hist_energy


def log_n_walkers(walkers, debug=True, limit=10):
    dprint = get_dprinter(debug)
    idx = np.argsort(abs(walkers))[::-1]
    sorted_arr = walkers[idx]
    mask = sorted_arr != 0

    idx = idx[mask]
    sorted_arr = sorted_arr[mask]

    for i, v in zip(idx[:limit], sorted_arr[:limit]):
      dprint(f"  {v} walkers on {i}th determinant")

    if sorted_arr.shape[0] > limit:
        dprint("  ... more")
    

def plot_result(E_fci, E_HF, hist_population, hist_shift, hist_energy):
    plt.figure(figsize=(10, 6))
    plt.plot(hist_population)
    plt.xlabel('Step')
    plt.ylabel('Population')
    plt.yscale('log')
    plt.grid(True)  

    E_corr = np.full_like(hist_shift, E_fci - E_HF)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_shift, label="Shift")
    plt.plot(E_corr, linestyle="--", label=r"${E_{corr}}$")
    plt.xlabel('Step')
    plt.ylabel('Shift')
    plt.legend()
    plt.grid(True)  

    E_fci_arr = np.full_like(hist_energy, E_fci)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_energy, label=r"${E_{proj}}$")
    plt.plot(E_fci_arr, linestyle="--", label=r"${E_{FCI}}$")
    plt.xlabel('Step')
    plt.ylabel('Projected Energy')
    plt.legend()
    plt.grid(True)  
    
    plt.show()