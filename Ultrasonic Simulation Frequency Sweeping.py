import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

C_WATER = 1480.0 #speed of sound in water (m/s)
rho = 997 # Density of water (kg/m^3)
sigma = 0.0728 # Surface tension of Water (N/m)
mu = 0.001 # Viscosity (Pa.s)
pv = 2338 # Vapor pressure of water (Pa)
pg0 = 10e5 # Initial gas pressure (Pa)
k = 1.4 # Polytropic Constant
pamb = 101325 # Ambient pressure (Pa)

R0 = 0.1e-3 # Initial Bubble Radius (m)
f = 10e3 # Ultrasound Frequency (Hz)

class Config: #Ultrasonic tank configuration

    Lx = 0.12             # Width(m)
    Ly = 0.07             # Height(m)
 
    Pdrive_atm   = 1.7    # Driving Pressure (Pa)
    Q            = 60     # Q Factor
 
    M_MAX = 8             # Highest mode number in x
    N_MAX = 8             # Highest mode number in y

    f_lo = 18_000.0        # bottom of the frequency band (Hz)
    f_hi = 50_000.0       # top of the frequency band    (Hz)
    fr = 21141 # resonsant frequency (Hz)


#Standing Wave Functions:
def mode_frequency(m, n, Lx, Ly): #Resonant frequency for mode (m, n)
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
        gamma = f0 / cfg.Q # width of resonance peak    
        denom = np.sqrt((f0 ** 2 - f ** 2) ** 2 + (gamma * f) ** 2) 
        r[(m, n)] = a * f0 ** 2 / denom 
    return r 


def response_scalar(f, cfg, modes):
    return max(mode_responses(f, cfg, modes).values())                                          

def reference_response(cfg, modes):
    fs = np.linspace(cfg.f_lo, cfg.f_hi, 1500)
    return max(response_scalar(f, cfg, modes) for f in fs)

 
def pressure_field(f, cfg, modes, aref, nx, ny): 

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

#Parameters

cfg = Config()

nx = 120
ny = 70

length = cfg.Lx          
width  = cfg.Ly 

#Bubble Distribution

bubble_density = 1e6   # nuclei per unit area              

R0_mean = 5.0e-6       # mean nucleus radius (m)            
R0_std  = (5/3)*1e-6   # std dev of nucleus radius (m)         

alpha = 1e-3
mu_growth = 0.01 / 60



rng = np.random.default_rng(seed=42)   
spacing = 1.0 / np.sqrt(bubble_density)      


N  = nx * ny

x_coords = (np.arange(nx) + 0.5) * spacing
y_coords = (np.arange(ny) + 0.5) * spacing


X, Y = np.meshgrid(x_coords, y_coords)
x = X.ravel()
y = Y.ravel()


R0 = rng.normal(R0_mean, R0_std, size=N)
R0[R0 < 0] = 0.0


bubbles = np.column_stack((x, y, R0))



def blake_threshold(R0):

    Pg0    = pamb - pv + 2 * sigma / R0
    PBlake = pamb - pv + (2 * sigma / (3 * R0)) * np.sqrt(3 / (Pg0 * R0 / (2 * sigma) + 1))

    return PBlake

def bacteria_dist(nx, ny, C0):
    C = np.ones((nx, ny)) * C0
    return C
    
C0 =1e4 #Initial Concentration of Bacteria CUF/mL

#The function below is adapted from the single bubble simulation to have R0 and f as input and to output rmax and rmin
def simulate_bubble(Pp, R0, f, treatment_time): #Gives Radius, Wall velocity and times for given Pp (peak pressure) and treatment time. The skeleton of this function was closely followed from an AI chat

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

   
    R = np.array(sol.y[0]) #saving radius
    

    rmin = np.min(R)
    rmax = np.max(R)
    return rmin, rmax

# Main Simulation 

modes = build_modes(cfg)
ar = reference_response(cfg, modes)
bac = bacteria_dist(nx, ny, C0)

def simul(f, t):
    # Defining variables
    modes = build_modes(cfg)
    ar = reference_response(cfg, modes)
    cfw = C0 * np.exp(mu_growth * t) #concentration of untreated water #change 600 to whatever time value
    C_final = np.exp(mu_growth * t) *bac.copy() #store concentration after treatment at different points
    
    xp, yp, field, pp = pressure_field(f, cfg, modes, ar, nx, ny)

    grid = [] #Build grid to evaluate each point in water tank
    for i in range(nx):
        for j in range(ny):
            grid.append([i, j])
   

    cav = [] #stores cavitation events

    for point in grid:
        x, y = point[0], point[1]
        
        R0 = bubbles[grid.index(point)][2]
        b = bac[x][y]
        
        p_a = abs(field[y][x] * 101325)
        
        if R0 > 0: #Skips points with no bubbles
                       
            # Calculate Blake threshold
            bt_pa = blake_threshold(R0)
            
            
            if p_a > bt_pa: # Check if pressure exceeds Blake threshold (cavitation occurs)
                    # Run RP simulation

                    rmin, rmax = simulate_bubble(p_a, R0, f, t)
                    if rmax > 2 * R0 and rmin < R0: #checks cavitaiton criteria
                        
                        k = alpha*(rmax/R0)**3 #kill rate scaled by R^3
                        cav.append([x, y, k, b])  # Store grid coordinates, kill rate, and conc. bacteria


    for event in cav: #goes through each cavitation event
        k = event[2] #kill reate at that event 
        C = event[3] #conc. bacteria at that event
        net_rate = mu_growth - k #overall growth rate 
        C = C * np.exp(net_rate * t) #calculate conc. bacteria
        C = np.maximum(C, 0.0) #a negative number is nonsensical physically, so make it 0 if engative
        C_final[event[0]][event[1]] = C
    
    cf = np.mean(C_final) #conc. of bacteria with treatment
    return cf, cfw

bw = cfg.fr/(cfg.Q*2) #calculate bandwidth of resonant frequency

Ct = [] #concentraiton values
kb = [] # %bacteria killed values


def simul_sweep(f, t, Cf, Cc): #ammended simul function to take the concentration of previous frequency treatment 
        # Defining variables
    modes = build_modes(cfg)
    ar = reference_response(cfg, modes)
    C_final = Cf.copy()
    
    xp, yp, field, pp = pressure_field(f, cfg, modes, ar, nx, ny)

    grid = [] #Build grid to evaluate each point in water tank
    for i in range(nx):
        for j in range(ny):
            grid.append([i, j])
   

    cav = [] #stores cavitation events

    for point in grid:
        x, y = point[0], point[1]
        
        R0 = bubbles[grid.index(point)][2]
        b = bac[x][y]
        
        p_a = abs(field[y][x] * 101325)
        
        if R0 > 0: #Skips points with no bubbles
                       
            # Calculate Blake threshold
            bt_pa = blake_threshold(R0)
            
            
            if p_a > bt_pa: # Check if pressure exceeds Blake threshold (cavitation occurs)
                    # Run RP simulation

                    rmin, rmax = simulate_bubble(p_a, R0, f, t)
                    if rmax > 2 * R0 and rmin < R0: #checks cavitaiton criteria
                        
                        k = alpha*(rmax/R0)**3 #kill rate scaled by R^3
                        cav.append([x, y, k, b])  # Store grid coordinates, kill rate, and conc. bacteria
    
    for event in cav:# goes through point where cavitaiton occurs
        k = event[2] #kill reate at that time
        C = event[3] #no. bacter at that time
        net_rate = mu_growth - k #overall growth rate 
        C = Cc[event[0]][event[1]] * np.exp(net_rate * t) #calculate no. bacteria
        C = np.maximum(C, 0.0) #a negative number is nonsensical physically, so make it 0 if engative
        C_final[event[0]][event[1]] = C
    
    
    
    return C_final

def freq_sweep(fr, bw, n, t_step): #sweeps n frequencies accross bandwidth bw of res freq fr for t_step seconds for each freq

    C_untreated = C0 * np.exp(mu_growth * t_step*n) #concentration of untreated water
    freq_step = 2 * bw / n     
    freqs = np.linspace(fr-bw,  fr + bw, n)
    
    Ccurrent = bacteria_dist(120, 70, C0)  # start from initial bacteria
   
    for freq in freqs:
        Cgrown = Ccurrent * np.exp(mu_growth * t_step) # how much bacteria would grow without treatment in this time interval
        Cnext = simul_sweep(freq, t_step, Cgrown, Ccurrent) #concentration after treatment
        Ccurrent = Cnext #set the current concentration to cnext for the next frequency sweep
        C_treated = np.mean(Ccurrent) #numerical value of concentration after treatment
        kbac = (1-C_treated/C_untreated)*100 #% bacteria killed
        print(f"{freq} Hz swept")
    return kbac

print(freq_sweep(cfg.fr, bw, 5, 600))
