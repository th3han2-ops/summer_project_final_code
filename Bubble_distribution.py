import numpy as np
import matplotlib.pyplot as plt

length = 0.10          
width  = 0.10
#arbritary parameters

bubble_density = 1e6   

R0_mean = 5.0e-6       
R0_std  = (5/3)*1e-6  
#for gaussian distribution

rng = np.random.default_rng(seed=42) #those who know

spacing = 1.0 / np.sqrt(bubble_density)    
nx = int(np.floor(length / spacing))
ny = int(np.floor(width  / spacing))
N  = nx * ny
x_coords = (np.arange(nx) + 0.5) * spacing
y_coords = (np.arange(ny) + 0.5) * spacing
X, Y = np.meshgrid(x_coords, y_coords)
x = X.ravel()
y = Y.ravel()

R0 = rng.normal(R0_mean, R0_std, size=N)
R0[R0 < 0] = 0.0
#gaussian distribution for radius, neg values give 0

bubbles = np.column_stack((x, y, R0))
print(f"Spacing between bubbles : {spacing*1e3:.3f} mm")
print(f"Grid                    : {nx} x {ny} = {N} bubbles")
print(f"Requested (rho*area)    : {bubble_density*length*width:.0f} bubbles")
print(f"R0 [um]  mean={R0.mean()*1e6:.2f}  std={R0.std()*1e6:.2f}  "
      f"min={R0.min()*1e6:.2f}  max={R0.max()*1e6:.2f}")
print(f"Empty sites (R0 = 0)    : {np.count_nonzero(R0 == 0)} of {N}")
np.savetxt("bubbles.csv", bubbles, delimiter=",",
           header="x_m,y_m,R0_m", comments="")
#putting in array 

plt.figure(figsize=(6, 5))
sc = plt.scatter(bubbles[:, 0]*1e3, bubbles[:, 1]*1e3,
                 c=bubbles[:, 2]*1e6, s=6, cmap="viridis")
plt.colorbar(sc, label=r"$R_0$  [$\mu$m]")
plt.xlabel("x  [mm]")
plt.ylabel("y  [mm]")
plt.title("Bubble nuclei: positions (grid) and radii (Gaussian)")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
#graphing
