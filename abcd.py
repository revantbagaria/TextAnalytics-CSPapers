import matplotlib.pyplot as plt
import numpy as np

a = [1,2,3,5,6,7,8]
a = np.array(a)

fig, ax = plt.subplots(figsize=(10,5)) 
ax.plot(a)
plt.show()