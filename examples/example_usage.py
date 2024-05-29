import mavrick as mv
import numpy as np

# Load an image
image = mv.load_image('../mavrick/resources/sample_images/test.jpg')

# Apply the Gaussian filter to the image
filtered_image = mv.gaussian_filter(image, size=25, sigma=10)
# Apply the Grayscale filter to the image
grayscale_image = mv.grayscale_filter(image)

# Save the filtered image
mv.save_image(filtered_image, '../mavrick/resources/sample_images/filtered_test.jpg')
# Save the grayscale image
mv.save_image(grayscale_image, '../mavrick/resources/sample_images/grayscale_test.jpg')
