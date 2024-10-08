import numpy as np
import multilayer_perceptron as mlp

multilayerPerceptron = mlp.MultiLayerPerceptron()


input_data1 = np.array([0.33,0.58])
output_data1 = multilayerPerceptron.forward(input_data1)
print(f"Output 1: {output_data1}")  # Output 1: [0.8580292  1.31019328]


input_data2 = np.array([0.13,0.96])
output_data2 = multilayerPerceptron.forward(input_data2)
print(f"Output 2: {output_data2}")  # Output 2: [0.86543634 1.32092095]


input_data3 = np.array([0.59,0.9])
output_data3 = multilayerPerceptron.forward(input_data3)
print(f"Output 3: {output_data3}")  # Output 3: [0.86881342 1.32575527]
