import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, ifft, rfftfreq

velocities = np.arange(0, 2, 0.001)
alpha = 0.5
f_a_05 = []
for velocity in velocities:
    force = velocity ** alpha
    f_a_05.append(force)

alpha = 1
f_a_1 = []
for velocity in velocities:
    force = velocity ** alpha
    f_a_1.append(force)
alpha = 1.5

f_a_15 = []
for velocity in velocities:
    force = velocity ** alpha
    f_a_15.append(force)

alpha = 3
f_a_3 = []
for velocity in velocities:
    force = velocity ** alpha
    f_a_3.append(force)



plt.plot(velocities, f_a_05, label='a = 0.5')
plt.plot(velocities, f_a_1, label='a = 1.0')
plt.plot(velocities, f_a_15, label='a = 1.5')
plt.plot(velocities, f_a_3, label='a = 3')


plt.xlabel('Velocity')
plt.ylabel('Force')
plt.title('Force-Velocity curves for different dashpots')
plt.legend()
plt.show()

