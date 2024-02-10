import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, ifft, rfftfreq

def mass_spring_dashpot_system(m, c, k, F0, dt, num_steps, alpha):
    # Initialize arrays to store results
    x_values = np.zeros(num_steps)
    t_values = np.zeros(num_steps)

    # Set initial conditions
    x = 1.0
    v = 0.0

    # Perform Euler integration
    for i in range(num_steps):
        x_values[i] = x
        t_values[i] = i * dt

        # Calculate acceleration using the differential equation
        a = (F0 - c * (((v))**alpha).real - k * x) / m

        # Update variables using Euler method
        v += a * dt
        x += v * dt

    return t_values, x_values

# System parameters
mass = 1.0
damping_coefficient = 0.5
spring_constant = 10.0
step_force_amplitude = 0

# Simulation parameters
time_step = 0.001
num_steps = 100000

t = np.arange(0.0,num_steps*time_step,time_step)

peaks = []
alphas = []
bandwidths = []
for alpha in range(50, 150):
# Perform simulation
    t_values, x_values = mass_spring_dashpot_system(mass, damping_coefficient, spring_constant, step_force_amplitude, time_step, num_steps, alpha/100)
    yf = rfft(x_values)
    xf = rfftfreq(num_steps, 0.01)*100
    yf[0] = 0
    peak_val = max(yf)
    peaks.append(peak_val)
    above_thres = []
    for i in range(len(xf)):
        if yf[i] >= (peak_val/(2**0.5)):
            above_thres.append(xf[i])
    bandwidth = max(above_thres) - min(above_thres)
    bandwidths.append(bandwidth)
    alphas.append(alpha/100)

plt.plot(alphas, np.abs(bandwidths))


plt.xlabel('Alpha.')
plt.ylabel('Half power bandwidth')
plt.title('Mass-Spring-Dashpot System Response to Step Input')
plt.legend()
plt.show()

