import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#constants
rho   = 998.0      
sigma = 0.0728     #surface tension
mu    = 1e-3       #dynamic viscosity
gamma = 1.4        #adiabatic index

p_inf = 101325.0   
p_v   = 2338.0     

#driving parameters
f_drive = 40e3     # 40 kHz (adjust to your transducer frequency)
P_a     = 1.5 * p_inf   # acoustic pressure amplitude (1.5 atm)

#integration parameters
n_cycles = 5       
t_span = (0, n_cycles / f_drive)
t_eval_dense = np.linspace(0, n_cycles / f_drive, 5000)


#acoustic pressure stuff
def p_acoustic(t):
    return p_inf - P_a * np.sin(2 * np.pi * f_drive * t)


#single bubble solver
def solve_bubble(R0_val, p_g0_val):
    def rp_rhs(t, y):
        R, Rdot = y
        
        if R <= 0:
            return [Rdot, 0.0]
        p_g = p_g0_val * (R0_val / R) ** (3 * gamma)
        p_ext = p_acoustic(t)
        pressure_term = (p_g - p_ext) / rho
        surface_term  = -2 * sigma / (rho * R)
        viscous_term  = -4 * mu * Rdot / (rho * R)
        kinetic_term  = -1.5 * Rdot ** 2
        Rddot = (pressure_term + surface_term + viscous_term + kinetic_term) / R
        return [Rdot, Rddot]
    
    sol = solve_ivp(
        rp_rhs,
        t_span,
        [R0_val, 0.0],
        method='LSODA',
        t_eval=t_eval_dense,
        rtol=1e-8,
        atol=1e-12,
        dense_output=True
    )
    
    return sol


#detecting cavitation
def detect_cavitation(sol, R0_val):
    R_traj = sol.y[0]
    max_idx, _ = find_peaks(R_traj)
    min_idx, _ = find_peaks(-R_traj)
    extrema = []
    for i in max_idx:
        extrema.append({"time": sol.t[i], "radius": R_traj[i], "type": "max", "idx": i})
    for i in min_idx:
        extrema.append({"time": sol.t[i], "radius": R_traj[i], "type": "min", "idx": i})
    extrema.sort(key=lambda x: x["time"])
    cavitation_events = []
    for i, ext in enumerate(extrema):
        if ext["type"] == "max" and ext["radius"] > 2 * R0_val:
            # Look for a subsequent minimum that dips below R0
            for j in range(i + 1, len(extrema)):
                if extrema[j]["type"] == "min" and extrema[j]["radius"] < R0_val:
                    cavitation_events.append(extrema[j])
                    break
    return cavitation_events


#setting bubble distribution
n_samples = 150
R0_mean = 5e-6      
R0_std = (5/3) * 1e-6  
R0_samples = np.random.normal(R0_mean, R0_std, n_samples)
R0_samples = np.abs(R0_samples[R0_samples > 0.1e-6])  
R0_samples = R0_samples[R0_samples < 100e-6] 
n_samples = len(R0_samples)
print(f"Ensemble size: {n_samples} bubbles")
print(f"R₀ range: {R0_samples.min()*1e6:.2f} – {R0_samples.max()*1e6:.2f} µm")
print(f"R₀ mean: {np.mean(R0_samples)*1e6:.2f} µm")

#solving bubbles
solutions = []
all_cavitation_events = []
max_radii = []
print("\nIntegrating ensemble...")
for i, R0_val in enumerate(R0_samples):
    p_g0_val = p_inf - p_v + 2 * sigma / R0_val
    try:
        sol = solve_bubble(R0_val, p_g0_val)
        if sol.status == 0:  # Success
            solutions.append((R0_val, sol))
            max_radii.append(np.max(sol.y[0]))
            cav_events = detect_cavitation(sol, R0_val)
            if cav_events:
                for event in cav_events:
                    all_cavitation_events.append({
                        'R0': R0_val,
                        'time': event['time'],
                        'radius': event['radius']
                    })
        else:
            print(f"  Warning: R₀={R0_val*1e6:.2f} µm integration failed")
    except Exception as e:
        print(f"  Error at R₀={R0_val*1e6:.2f} µm: {e}")
    if (i + 1) % 30 == 0:
        print(f"  ... {i+1}/{n_samples} complete")
print(f"\nSuccessful integrations: {len(solutions)}/{n_samples}")
print(f"Cavitation events detected: {len(all_cavitation_events)}")

#percentiles 
t_common = np.linspace(0, n_cycles / f_drive, 1000)
radius_ensemble = np.zeros((len(solutions), len(t_common)))

for i, (R0_val, sol) in enumerate(solutions):
    radius_ensemble[i] = sol.sol(t_common)[0]

p5 = np.percentile(radius_ensemble, 5, axis=0)
p25 = np.percentile(radius_ensemble, 25, axis=0)
p50 = np.percentile(radius_ensemble, 50, axis=0)
p75 = np.percentile(radius_ensemble, 75, axis=0)
p95 = np.percentile(radius_ensemble, 95, axis=0)


#following code was generated with the aid of ai. Use primarily for the plots. 


#plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#ensemble statistics
ax = axes[0, 0]

n_plot = min(20, len(solutions))
idx_plot = np.linspace(0, len(solutions) - 1, n_plot, dtype=int)

for idx in idx_plot:
    R0_val, sol = solutions[idx]
    R_plot = sol.sol(t_common)[0] * 1e6  # µm
    ax.plot(
        t_common * 1e6,
        R_plot,
        alpha=0.35,
        lw=1
    )

# percentile bands
ax.fill_between(
    t_common * 1e6,
    p5 * 1e6,
    p95 * 1e6,
    alpha=0.20,
    label='5–95 percentile'
)

ax.fill_between(
    t_common * 1e6,
    p25 * 1e6,
    p75 * 1e6,
    alpha=0.35,
    label='25–75 percentile'
)

# median
ax.plot(
    t_common * 1e6,
    p50 * 1e6,
    lw=2.5,
    label='Median'
)

# cavitation threshold
ax.axhline(
    2 * R0_mean * 1e6,
    color='red',
    linestyle='--',
    linewidth=2,
    label=r'$2R_{0,\mathrm{mean}}$'
)

ax.set_title('Bubble Radius Evolution')
ax.set_xlabel('Time (µs)')
ax.set_ylabel('Radius (µm)')
ax.legend()
ax.grid(True, alpha=0.3)

#radius distribution, second plot
ax = axes[0, 1]

R0_um = R0_samples * 1e6

ax.hist(
    R0_um,
    bins=20,
    alpha=0.75,
    edgecolor='black'
)

mean_R0 = np.mean(R0_um)
median_R0 = np.median(R0_um)

ax.axvline(
    mean_R0,
    linestyle='--',
    linewidth=2,
    label=f'Mean = {mean_R0:.2f} µm'
)

ax.axvline(
    median_R0,
    linestyle=':',
    linewidth=2,
    label=f'Median = {median_R0:.2f} µm'
)

ax.set_title(
    rf'$R_0 \sim \mathcal{{N}}({R0_mean*1e6:.2f},\,{R0_std*1e6:.2f}^2)\ \mu m$'
)
ax.set_xlabel(r'Initial Radius $R_0$ (µm)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

#leaving bottom panels blank
axes[1, 0].axis('off')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
