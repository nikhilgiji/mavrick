import mavrick as mv
import numpy as np

# Load an image
image = mv.load_image('../mavrick/resources/sample_images/test.jpg')

# Apply the Gaussian filter to the image
filtered_image = mv.gaussian_filter(image, size=25, sigma=10)

# Save the filtered image
mv.save_image(filtered_image, '../mavrick/resources/sample_images/filtered_test.jpg')
