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
time_step = 0.01
num_steps = 10000

t = np.arange(0.0,num_steps*time_step,time_step)

# Perform simulation
t_values, x_values = mass_spring_dashpot_system(mass, damping_coefficient, spring_constant, step_force_amplitude, time_step, num_steps, 0.5)

# Plot the results
yf = rfft(x_values)
xf = rfftfreq(num_steps, 0.1)*10
yf[0] = 0
#plt.plot(xf, np.abs(yf),label='fft a = 0.5')

plt.plot(t_values, x_values, label='a = 0.5')

t_values, x_values = mass_spring_dashpot_system(mass, damping_coefficient, spring_constant, step_force_amplitude, time_step, num_steps, 1)
yf = rfft(x_values)
xf = rfftfreq(num_steps, 0.1)*10
yf[0] = 0
#plt.plot(xf, np.abs(yf),label='fft a = 1')
# Plot the results
plt.plot(t_values, x_values, label='a = 1')
t_values, x_values = mass_spring_dashpot_system(mass, damping_coefficient, spring_constant, step_force_amplitude, time_step, num_steps, 1.1)

# Plot the results


yf = rfft(x_values)
xf = rfftfreq(num_steps, 0.1)*10
yf[0] = 0
#plt.plot(xf, np.abs(yf),label='fft a = 1.1')
plt.plot(t_values, x_values, label='a = 1.1')


t_values, x_values = mass_spring_dashpot_system(mass, damping_coefficient, spring_constant, step_force_amplitude, time_step, num_steps, 3)

# Plot the results


yf = rfft(x_values)
xf = rfftfreq(num_steps, 0.1)*10
yf[0] = 0
#plt.plot(xf, np.abs(yf),label='fft a =3')
plt.plot(t_values, x_values, label='a = 3')


plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Mass-Spring-Dashpot System Response to Impulse')
plt.legend()
plt.show()

