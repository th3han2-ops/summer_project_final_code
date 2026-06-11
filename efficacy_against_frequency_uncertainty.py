
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


C_WATER = 1480.0          
rho     = 998.0           
sigma   = 0.0728          #surface tension
mu      = 1e-3            #dynamic viscosity 
gamma   = 1.4             #adiabatic index 
p_amb   = 101325.0       
p_v     = 2338.0          
#constants

class Config:
    Lx = 0.12             
    Ly = 0.07             
    Pdrive_atm = 1.7  #atm not pressure    
    Q          = 60      
    M_MAX = 8             
    N_MAX = 8            
    f_lo  = 18_000.0      
    f_hi  = 50_000.0      
#tank and transducer params

cfg = Config()

alpha     = 1e-3          # kill-rate constant
mu_growth = 0.01 / 60     # bacterial growth rate
fr        = 21141.4       
C0        = 1e4           # initial contaminant concentration 
#bacteria params

DEFAULT = dict(
    Q=60.0,
    Pdrive_atm=1.7,   Pdrive_std=0.2,
    alpha=1e-3,
    R0_mean=5.0e-6,   R0_std=(5/3)*1e-6,
    mu_growth=0.01/60, C0=1e4, t=600.0,
)
#parameter dictionary

#standing wave stuff
def mode_frequency(m, n, Lx, Ly):
    return (C_WATER / 2.0) * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)

def build_modes(cfg):
    modes = []
    for m in range(cfg.M_MAX + 1):
        for n in range(cfg.N_MAX + 1):
            if m == 0 and n == 0:
                continue
            f0 = mode_frequency(m, n, cfg.Lx, cfg.Ly)
            a = 1.0 / (1.0 + m + n)
            modes.append((m, n, f0, a))
    return modes

def mode_responses(f, cfg, modes):
    r = {}
    for (m, n, f0, a) in modes:
        g = f0 / cfg.Q
        denom = np.sqrt((f0 ** 2 - f ** 2) ** 2 + (g * f) ** 2) 
        r[(m, n)] = a * f0 ** 2 / denom 
    return r 
def response_scalar(f, cfg, modes):
    return max(mode_responses(f, cfg, modes).values())
def reference_response(cfg, modes):
    fs = np.linspace(cfg.f_lo, cfg.f_hi, 1500)
    return max(response_scalar(f, cfg, modes) for f in fs)
    
#pressure field from standin waves
def pressure_field(f, cfg, modes, aref, nx=240, ny=140):

    u = np.linspace(0, 1, nx) 
    v = np.linspace(0, 1, ny)
    U, V = np.meshgrid(u, v)
    r = mode_responses(f, cfg, modes)
    S = np.zeros_like(U) 
    for (m, n, f0, a) in modes:
        S += r[(m, n)] * np.cos(m * np.pi * U) * np.cos(n * np.pi * V)
    Smax = np.max(np.abs(S)) 
    Ppeak = cfg.Pdrive_atm * response_scalar(f, cfg, modes) / aref 
    field = Ppeak * S / Smax if Smax > 0 else S
    X = U * cfg.Lx * 100.0                    
    Y = V * cfg.Ly * 100.0
    return X, Y, field, Ppeak
 
 
#blacke threshold and Rayleigh-Plesset through LSODA
def simulate_bubble(P_a, R0, f, n_cycles):
    def p_acoustic_local(t):
        return p_amb - P_a * np.sin(2 * np.pi * f * t)

    def rp_rhs_local(t, y):
        R, Rdot = y
        if R <= 0:
            return [Rdot, 0.0]
        p_g0 = p_amb - p_v + 2 * sigma / R0
        p_g = p_g0 * (R0 / R) ** (3 * gamma)
        p_ext = p_acoustic_local(t)
        Rddot = ((p_g - p_ext) / rho - 2 * sigma / (rho * R)
                 - 4 * mu * Rdot / (rho * R) - 1.5 * Rdot ** 2) / R
        return [Rdot, Rddot]

    t_span = (0, n_cycles / f)
    t_eval = np.linspace(*t_span, 5000)
    sol = solve_ivp(rp_rhs_local, t_span, [R0, 0.0], method='LSODA',
                    t_eval=t_eval, rtol=1e-8, atol=1e-12)
    return np.min(sol.y[0]), np.max(sol.y[0])

def blake_threshold(R0):
    P_g0 = p_amb - p_v + 2 * sigma / R0
    return p_amb - p_v + (2 * sigma / (3 * R0)) * np.sqrt(
        3 / (P_g0 * R0 / (2 * sigma) + 1))


#MAIN STUFF HERE
def make_cfg(Q, Pdrive_atm):
    c = Config(); c.Q = Q; c.Pdrive_atm = Pdrive_atm
    return c

def make_nuclei(seed, nx, ny, R0_mean, R0_std):
    rng = np.random.default_rng(seed)
    R = rng.normal(R0_mean, R0_std, size=(ny, nx))
    R[R < 0] = 0.0
    return R
    #bubble nuclei gaussian stuff

def run_sweep(freqs, seed, nx, ny, params, n_cycles=5):
    c = make_cfg(params["Q"], params["Pdrive_atm"])
    modes = build_modes(c)
    aref = reference_response(c, modes)
    R0_grid = make_nuclei(seed, nx, ny, params["R0_mean"], params["R0_std"])
    a_kill = params["alpha"]
    mu_g   = params["mu_growth"]
    C0_    = params["C0"]
    t      = params["t"]
    cfw    = C0_ * np.exp(mu_g * t)
    
    eff = np.empty(len(freqs))
    for fi, f in enumerate(freqs):
        _, _, field, _ = pressure_field(f, c, modes, aref, nx, ny)
        C_final = np.full((ny, nx), cfw)
        for j in range(ny):
            for i in range(nx):
                R0 = R0_grid[j, i]
                if R0 <= 0:
                    continue
                p_a = abs(field[j, i] * 101325.0)             # atm -> Pa
                if p_a <= blake_threshold(R0):
                    continue
                rmin, rmax = simulate_bubble(p_a, R0, f, n_cycles)
                if rmax > 2 * R0 and rmin < R0:               
                    k = a_kill * (rmax / R0) ** 3
                    C_final[j, i] = C0_ * np.exp((mu_g - k) * t)
        eff[fi] = cfw / np.mean(C_final)
    return eff
#retruns efficacy 


def lorentzian(f, base, A, f0, g):
    return base + A / (1.0 + ((f - f0) / g) ** 2)

def fit_peak(freqs, mean, se):
    p0 = [mean.min(), mean.max() - mean.min(),
          freqs[np.argmax(mean)], (freqs[-1] - freqs[0]) / 4]
    s = np.maximum(se, 1e-12)
    popt, pcov = curve_fit(lorentzian, freqs, mean, p0=p0,
                           sigma=s, absolute_sigma=True, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr
#fitting a lorentzian curve

#the following code was written with the aid of generative ai. 
N_SEEDS  = 200
NX, NY   = 20, 10
N_FREQS  = 15
N_CYCLES = 5

params = DEFAULT.copy()

# Frequency array
# fr ± fr/(2Q)

Q = params["Q"]
half_band = fr / (2.0 * Q)
freqs = np.linspace(
    fr - half_band,
    fr + half_band,
    N_FREQS
)

#running the seeds
all_runs = []
t0 = time.time()
for seed in range(N_SEEDS):
    print(f"Seed {seed+1}/{N_SEEDS}")
    eff = run_sweep(
        freqs=freqs,
        seed=seed,
        nx=NX,
        ny=NY,
        params=params,
        n_cycles=N_CYCLES
    )
    all_runs.append(eff)
all_runs = np.vstack(all_runs)
print(f"\nCompleted in {(time.time()-t0)/60:.2f} min")

#saving results
np.savez(
    "removal_runs.npz",
    freqs=freqs,
    efficacy=all_runs
)

#converting efficacy to removal efficiency
removal = (1.0 - 1.0 / all_runs) * 100.0

N = removal.shape[0]

mean_removal = np.mean(removal, axis=0)

sd_removal = np.std(
    removal,
    axis=0,
    ddof=1
)

se_removal = sd_removal / np.sqrt(N)

# Lorentzian fit
popt, perr = fit_peak(
    freqs,
    mean_removal,
    se_removal
)

base, A, f0_fit, g_fit = popt
base_err, A_err, f0_err, g_err = perr

# Fine grid for smooth fitted curve
freq_fine = np.linspace(
    freqs.min(),
    freqs.max(),
    400
)
fit_curve = lorentzian(
    freq_fine,
    *popt
)
peak_removal = lorentzian(
    f0_fit,
    *popt
)

#plot
plt.figure(figsize=(8, 5))

# ±SD shaded region
plt.fill_between(
    freqs,
    mean_removal - sd_removal,
    mean_removal + sd_removal,
    alpha=0.25,
    label="±1 SD"
)

# Mean ± SE
plt.errorbar(
    freqs,
    mean_removal,
    yerr=se_removal,
    fmt='o',
    capsize=3,
    label="Mean ± SE"
)

# Lorentzian fit
plt.plot(
    freq_fine,
    fit_curve,
    'k--',
    linewidth=2,
    label="Lorentzian fit"
)

# Peak frequency marker
plt.axvline(
    f0_fit,
    color='k',
    linestyle=':',
    linewidth=1.5
)

# Annotation
plt.annotate(
    rf"$f_0 = {f0_fit:.1f} \pm {f0_err:.1f}\,\mathrm{{Hz}}$",
    xy=(f0_fit, peak_removal),
    xytext=(10, 15),
    textcoords="offset points"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Removal (%)")
plt.title("Removal Efficiency vs Frequency (200 Seeds)")
plt.legend()
plt.tight_layout()

plt.savefig(
    "fig_efficacy_200_seeds.png",
    dpi=150
)

plt.show()


#summary
print("\n====================================")
print(f"Peak removal = {peak_removal:.3f} %")
print(f"Fitted f0    = {f0_fit:.2f} ± {f0_err:.2f} Hz")
print("====================================")
