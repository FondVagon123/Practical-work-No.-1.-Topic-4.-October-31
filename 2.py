import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz


n_points = 100

x1_range = np.linspace(-7, 3, n_points) 
x2_range = np.linspace(-4.4, 1.7, n_points)

X1, X2 = np.meshgrid(x1_range, x2_range)




def calculate_sugeno_components(x1_range, x2_range):
    
    x1_low = fuzz.trimf(x1_range, [-7, -7, -5])
    x1_medium = fuzz.trimf(x1_range, [-6, -2, 2])
    x1_high = fuzz.trimf(x1_range, [0, 3, 3])

    
    x2_low = fuzz.trimf(x2_range, [-4.4, -4.4, -2])
    x2_medium = fuzz.trimf(x2_range, [-3, -1, 1])
    x2_high = fuzz.trimf(x2_range, [0, 1.7, 1.7])

    x1_mfs = {'low': x1_low, 'medium': x1_medium, 'high': x1_high}
    x2_mfs = {'low': x2_low, 'medium': x2_medium, 'high': x2_high}

   
    consequents = {
        'y0': lambda x1, x2: 0,
        'y50': lambda x1, x2: 50,
        'y4x1_m1x2': lambda x1, x2: 4*x1 - 1*x2 + 0,
        'y2x1_2x2_1': lambda x1, x2: 2*x1 + 2*x2 + 1,
        'y8x1_2x2_8': lambda x1, x2: 8*x1 + 2*x2 + 8,
    }

   
    rules = [
        ('medium', None, 'y0'),
        ('high', 'high', 'y2x1_2x2_1'),
        ('high', 'low', 'y4x1_m1x2'),
        ('low', 'medium', 'y8x1_2x2_8'),
        ('low', 'low', 'y50'),
        ('low', 'high', 'y50'),
    ]
    return x1_mfs, x2_mfs, consequents, rules


def sugeno_inference(X1, X2, x1_range, x2_range):
    x1_mfs, x2_mfs, consequents, rules = calculate_sugeno_components(x1_range, x2_range)
    total_output = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x1_val, x2_val = X1[i, j], X2[i, j]
            rule_weights, rule_outputs = [], []

            
            x1_mfx = {k: fuzz.interp_membership(x1_range, v, x1_val) for k, v in x1_mfs.items()}
            x2_mfx = {k: fuzz.interp_membership(x2_range, v, x2_val) for k, v in x2_mfs.items()}

            for x1_term, x2_term, conseq_key in rules:
                
                weight = 1.0 
                if x1_term: weight = min(weight, x1_mfx.get(x1_term, 0))
                if x2_term: weight = min(weight, x2_mfx.get(x2_term, 0))

                
                output = consequents[conseq_key](x1_val, x2_val)

                rule_weights.append(weight)
                rule_outputs.append(output)

            weights_arr, outputs_arr = np.array(rule_weights), np.array(rule_outputs)
            sum_weights = np.sum(weights_arr)
            
            
            if sum_weights == 0:
                y_out = np.mean(outputs_arr) if outputs_arr.size > 0 else 0
            else:
                y_out = np.sum(weights_arr * outputs_arr) / sum_weights

            total_output[i, j] = y_out

    return total_output


Y_sugeno = sugeno_inference(X1, X2, x1_range, x2_range)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y_sugeno, cmap='jet')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('2. Поверхня відгуку - Sugeno FIS')
plt.show()
