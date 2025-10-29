import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz


n_points = 100
x1_range = np.linspace(-7, 3, n_points)
x2_range = np.linspace(-4.4, 1.7, n_points)
X1, X2 = np.meshgrid(x1_range, x2_range)


y_universe = np.linspace(-25, 60, 200)


def calculate_sugeno_components(x1_range, x2_range):
    
    x1_low = fuzz.trimf(x1_range, [-7, -7, -5])
    x1_medium = fuzz.trimf(x1_range, [-6, -2, 2])
    x1_high = fuzz.trimf(x1_range, [0, 3, 3])

    
    x2_low = fuzz.trimf(x2_range, [-4.4, -4.4, -2])
    x2_medium = fuzz.trimf(x2_range, [-3, -1, 1])
    x2_high = fuzz.trimf(x2_range, [0, 1.7, 1.7])

    x1_mfs = {'low': x1_low, 'medium': x1_medium, 'high': x1_high}
    x2_mfs = {'low': x2_low, 'medium': x2_medium, 'high': x2_high}

    return x1_mfs, x2_mfs, None, None 




y_mfs_mamdani = {
    'vvlow': fuzz.trimf(y_universe, [-25, -25, -10]), 
    'low': fuzz.trimf(y_universe, [-15, 0, 15]),     
    'medium': fuzz.trimf(y_universe, [0, 15, 30]),   
    'high': fuzz.trimf(y_universe, [20, 35, 50]),    
    'vhigh': fuzz.trimf(y_universe, [45, 60, 60]),   
}


mamdani_rules = [
    
    ('medium', None, 'low'),
    ('high', 'high', 'medium'),
    ('high', 'low', 'medium'),
    ('low', 'medium', 'high'),
    ('low', 'low', 'vhigh'),
    ('low', 'high', 'vhigh'),
]


def mamdani_inference(X1, X2, x1_range, x2_range, y_universe, y_mfs_mamdani, mamdani_rules):
    x1_mfs, x2_mfs, _, _ = calculate_sugeno_components(x1_range, x2_range)
    Y_mamdani = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x1_val, x2_val = X1[i, j], X2[i, j]
            
            x1_mfx_val = {k: fuzz.interp_membership(x1_range, v, x1_val) for k, v in x1_mfs.items()}
            x2_mfx_val = {k: fuzz.interp_membership(x2_range, v, x2_val) for k, v in x2_mfs.items()}
            
            aggregated_output = np.zeros_like(y_universe)
            
            for x1_term, x2_term, y_term in mamdani_rules:
                
                weight = 1.0
                if x1_term: weight = min(weight, x1_mfx_val.get(x1_term, 0))
                if x2_term: weight = min(weight, x2_mfx_val.get(x2_term, 0))

                
                activated_mf = np.fmin(weight, y_mfs_mamdani[y_term])
                
               
                aggregated_output = np.fmax(aggregated_output, activated_mf)

            if np.sum(aggregated_output) == 0:
                y_out = np.mean(y_universe)
            else:
                y_out = fuzz.defuzz(y_universe, aggregated_output, 'centroid')
            
            Y_mamdani[i, j] = y_out

    return Y_mamdani


Y_mamdani = mamdani_inference(X1, X2, x1_range, x2_range, y_universe, y_mfs_mamdani, mamdani_rules)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y_mamdani, cmap='jet')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3. Поверхня відгуку - Mamdani FIS')
plt.show()
