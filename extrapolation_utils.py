import matplotlib.pyplot as plt
import numpy as np

def inverse_fit(N_det, E):
    x = 1.0 / N_det
    p, cov = np.polyfit(x, E, 1, cov=True)
    a, E_inf = p
    sd = np.sqrt(np.diag(cov))
    a_sd, E_sd = sd
    fn = lambda N_det: a/N_det + E_inf    
    return a, E_inf, fn, a_sd, E_sd

def plot_extrapolation_to_inf_det(N_det, E, E_sd, fn):    
    N_det_fit = np.linspace(N_det.min(), N_det.max(), 500)
    E_fit = fn(N_det_fit)

    # plt.scatter(N_det, E, label="data")
    plt.plot(N_det_fit, E_fit, label=r"fit: $E_\infty + a/N$")
    plt.errorbar(N_det, E, yerr=E_sd, fmt='o', capsize=3, label=r"$E_{est}$")
    plt.xlabel("Number of determinants")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()