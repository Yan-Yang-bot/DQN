import numpy as np
import matplotlib.pyplot as plt

data = np.load('to_plot.npz')

xlabel = [i*1000+5000 for i in range(96)]

plt.plot(xlabel, data)

plt.show()