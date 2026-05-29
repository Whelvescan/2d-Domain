import numpy as np
import matplotlib.pyplot as plt
n=30
x=np.linspace(.1,3,n)
f=np.sin(x)

dx = x[1] - x[0]
dfdx = np.zeros(n)

print("Starting array:", dfdx)

# Left Boundary
dfdx[0] = (f[1] - f[0]) / dx
print("After step 1 (Forward Diff):", dfdx)

# Interior Loop
for i in range(1, n - 1):
    dfdx[i] = (f[i + 1] - f[i - 1]) / (2 * dx)
print("After step 2 (Central Diff loop):", dfdx)

# Right Boundary
dfdx[-1] = (f[-1] - f[-2]) / dx
plt.figure
plt.plot(x,np.cos(x),'k',label='true der')
plt.plot(x,dfdx,'r',label='computed der',linewidth=2)
plt.legend()
plt.show()