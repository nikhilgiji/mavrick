import mavrick as mv
import numpy as np

# Load an image
image = mv.load_image('../mavrick/resources/sample_images/test.jpg')

# Apply the Grayscale filter to the image
grayscale_image = mv.grayscale_filter(image)

# Save the grayscale image
mv.save_image(grayscale_image, '../mavrick/resources/sample_images/grayscale_test.jpg')

# Apply the Brightness filter to the image
bright_image = mv.brightness_filter(image, factor=1.5)

# Save the brightened image
mv.save_image(bright_image, '../mavrick/resources/sample_images/bright_test.jpg')
