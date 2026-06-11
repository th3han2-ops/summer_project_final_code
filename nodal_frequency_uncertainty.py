
import numpy as np
import matplotlib.pyplot as plt

#params and uncertainties
C_WATER = 1480.0;  SIG_C  = 3.0
LX = 0.120;        SIG_LX = 0.001
LY = 0.070;        SIG_LY = 0.001
M_MAX, N_MAX = 8, 8
Q_FACTOR = 300
F_LO, F_HI = 20_000.0, 40_000.0

N_MC = 400          #monte carlo
N_FREQ = 800        
SEED = 42


#model
def mode_frequency(m, n, c, lx, ly):
    return (c / 2.0) * np.sqrt((m / lx) ** 2 + (n / ly) ** 2)

def build_modes(c, lx, ly):
    f0_list, a_list = [], []
    for m in range(M_MAX + 1):
        for n in range(N_MAX + 1):
            if m == 0 and n == 0:
                continue
            f0_list.append(mode_frequency(m, n, c, lx, ly))
            a_list.append(1.0 / (1.0 + m + n))
    return np.array(f0_list), np.array(a_list)

def response_curve(freqs, f0, a, q):
    gamma = f0[:, None] / q
    denom = np.sqrt((f0[:, None]**2 - freqs[None, :]**2)**2
                    + (gamma * freqs[None, :])**2)
    r = a[:, None] * f0[:, None]**2 / denom
    return r.max(axis=0)      

def sigma_analytic(m, n, c, lx, ly, sig_c, sig_lx, sig_ly):
    f = mode_frequency(m, n, c, lx, ly)
    A = (m / lx) ** 2
    B = (n / ly) ** 2
    AB = A + B
    var = (f / c * sig_c) ** 2 \
        + (f * A / (AB * lx) * sig_lx) ** 2 \
        + (f * B / (AB * ly) * sig_ly) ** 2
    return np.sqrt(var)


#curve
freqs = np.linspace(F_LO, F_HI, N_FREQ)
f0_nom, a_nom = build_modes(C_WATER, LX, LY)
resp_nom = response_curve(freqs, f0_nom, a_nom, Q_FACTOR)


#monte carlo stuff
rng = np.random.default_rng(SEED)
all_curves = np.zeros((N_MC, N_FREQ))

for i in range(N_MC):
    c_s  = rng.normal(C_WATER, SIG_C)
    lx_s = rng.normal(LX, SIG_LX)
    ly_s = rng.normal(LY, SIG_LY)
    f0_s, a_s = build_modes(c_s, lx_s, ly_s)
    all_curves[i] = response_curve(freqs, f0_s, a_s, Q_FACTOR)

mean_curve = all_curves.mean(axis=0)
std_curve  = all_curves.std(axis=0, ddof=1)


#horizontal error bars
peaks = []
for m in range(M_MAX + 1):
    for n in range(N_MAX + 1):
        if m == 0 and n == 0:
            continue
        f = mode_frequency(m, n, C_WATER, LX, LY)
        if F_LO <= f <= F_HI:
            sig = sigma_analytic(m, n, C_WATER, LX, LY, SIG_C, SIG_LX, SIG_LY)
            resp_here = response_curve(np.array([f]), f0_nom, a_nom, Q_FACTOR)[0]
            peaks.append((f, sig, resp_here, m, n))


#following section uses generative ai for plotting


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 8),
    sharex=True,
    constrained_layout=True
)


freqs_khz = freqs / 1000.0

#plot 1
ax1.fill_between(
    freqs_khz,
    mean_curve - std_curve,
    mean_curve + std_curve,
    color="lightblue",
    alpha=0.6,
    label=r"MC $\pm1\sigma$"
)

ax1.plot(
    freqs_khz,
    mean_curve,
    color="navy",
    lw=2,
    label="MC mean"
)

ax1.plot(
    freqs_khz,
    resp_nom,
    "--",
    color="red",
    lw=1.8,
    label="Nominal"
)

ax1.set_ylabel("Response")
ax1.set_title("Cavity Response with Uncertainty")
ax1.legend()
ax1.grid(True, alpha=0.3)

#plot 2
ax2.plot(
    freqs_khz,
    resp_nom,
    color="0.7",
    lw=1.5,
    label="Nominal response"
)

for f, sig, resp_here, m, n in peaks:
    ax2.errorbar(
        f / 1000.0,
        resp_here,
        xerr=sig / 1000.0,
        fmt="o",
        color="red",
        ecolor="red",
        capsize=3,
        markersize=4
    )

ax2.set_xlabel("Frequency (kHz)")
ax2.set_ylabel("Response")
ax2.set_title("Modal Frequencies with Horizontal Uncertainty Bars")
ax2.grid(True, alpha=0.3)

print(f"Number of peaks marked: {len(peaks)}")

plt.show()
