import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#Constants
rho = 997 # Density of water (kg/m^3)
sigma = 0.0728 # Surface tension of Water (N/m)
mu = 0.001 # Viscosity (Pa.s)
pv = 2338 # Vapor pressure of water (Pa)
pg0 = 10e5 # Initial gas pressure (Pa)
k = 1.4 # Polytropic Constant
pamb = 101325 # Ambient pressure (Pa)

R0 = 50e-6 # Initial Bubble Radius (m)
f = 40e3 # Ultrasound Frequency (Hz)

def simulate_bubble(Pp, treatment_time): #Gives Radius, Wall velocity and times for given Pp (peak pressure) and treatment time. The skeleton of this function was closely followed from an AI chat

    def pinf_calc(Pp, t): #P_infinity as a function of time
         
        return pamb - Pp * np.sin(2 * np.pi * f * t)  

    def rhs(t, y): #input for LSODA solver
        R, Rdot = y

        if R <= 0:
            return [Rdot, 0.0]
        
        pinf = pinf_calc(Pp, t)
        
        pg0  = pamb - pv + 2*sigma/R0

        pg = pg0 * (R0 / R)**(3 * k)
        

        Rddot = ((pg - pinf) / rho - 2 * sigma / (rho * R)  - 4 * mu * Rdot / (rho * R)  -1.5 * Rdot**2 ) / R

        return [Rdot, Rddot]

    t_span = (0,treatment_time)
    t_eval = np.linspace(*t_span, 5000)

    sol = solve_ivp(
        rhs,
        t_span,
        [R0, 0.0],
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-12
    )

    times = np.array(sol.t) #saving time
    R = np.array(sol.y[0]) #saving radius
    Rdot = np.array(sol.y[1]) #saving wall velocity
  

    return R, Rdot, times

R, Rdot, times = simulate_bubble(1.5*pamb, 5/f) #evaluate at driving pressure 1.5atm and evaluate over 5 cycles

plt.plot(times, R)
plt.ylabel("Radius (m)")
plt.xlabel("Times (s)")
plt.title("Radius vs Time f = 40Hz P = 1.5 atm")
plt.show() #Plot radius vs time
