# from mavrick.core.image_io import load_image, save_image
# from mavrick.filters import apply_filter 
# from mavrick.filters import gaussian_filter
# import numpy as np 

# # Load an image
# image = load_image('../mavrick/resources/sample_images/test.jpg')

# # Define a simple blur filter
# # filter_matrix = np.ones((3, 3)) / 9

# # # Apply the filter to the image
# # filtered_image = apply_filter(image, filter_matrix)

# # Apply the Gaussian filter to the image
# filtered_image = gaussian_filter(image, size=20, sigma=10)
# # Save the filtered image
# save_image(filtered_image, '../mavrick/resources/sample_images/filtered_test.jpg')

import mavrick as mv
import numpy as np

# Load an image
image = mv.load_image('../mavrick/resources/sample_images/test.jpg')

# Apply the Gaussian filter to the image
filtered_image = mv.gaussian_filter(image, size=15, sigma=10)

# Save the filtered image
mv.save_image(filtered_image, '../mavrick/resources/sample_images/filtered_test.jpg')
