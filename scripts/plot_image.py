import os
import numpy as np
import matplotlib.pyplot as plt

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_folder = os.path.join(parent_directory, 'data')

image_path = os.path.join(data_folder, 'sukhna_lake_true-color_2020-01-01.npy')

# Load the image from the file
loaded_image = np.load(image_path)

# Plot the image
plt.figure()
plt.imshow(loaded_image)
plt.axis('off')
plt.show()