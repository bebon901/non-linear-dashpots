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

        # Calculate acceleration
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
time_step = 0.01
num_steps = 10000

t = np.arange(0.0,num_steps*time_step,time_step)

# Define arrays to plot
peaks = []
alphas = []

# Go through varying alpha
for alpha in range(50, 150):
    # Perform simulation
    t_values, x_values = mass_spring_dashpot_system(mass, damping_coefficient, spring_constant, step_force_amplitude, time_step, num_steps, alpha/100)
    # Perform Fourier analysis on the results
    yf = rfft(x_values)
    xf = rfftfreq(num_steps, 0.1)*10
    # Remove peak at w = 0. Were getting a rogue peak at w = 0
    yf[0] = 0
    peaks.append(max(yf))
    alphas.append(alpha/100)

plt.plot(alphas, np.abs(peaks))

plt.xlabel('Alpha.')
plt.ylabel('Peak')
plt.title('Mass-Spring-Dashpot System Response to Step Input')
plt.legend()
plt.show()

