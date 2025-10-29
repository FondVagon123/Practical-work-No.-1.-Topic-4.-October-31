import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz


x1_range = np.linspace(-7, 3, 50)
x2_range = np.linspace(-4.4, 1.7, 50)


X1, X2 = np.meshgrid(x1_range, x2_range)

Y_target = X1**2 * np.sin(X2 - 1) - 2


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y_target, cmap='jet')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('1. Еталонна поверхня (Target Surface)')
plt.show()
