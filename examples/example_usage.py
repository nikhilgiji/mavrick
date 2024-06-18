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

# Apply the Contrast filter to the image
contrast_image = mv.contrast_filter(image, factor=1.5)

# Save the contrast-adjusted image
mv.save_image(contrast_image, '../mavrick/resources/sample_images/contrast_test.jpg')

# Resize the image
resized_image = mv.resize_image(image, (200, 200))
mv.save_image(resized_image, '../mavrick/resources/sample_images/resized_test.jpg')

# Crop the image
cropped_image = mv.crop_image(image, top=50, left=50, height=100, width=100)
mv.save_image(cropped_image, '../mavrick/resources/sample_images/cropped_test.jpg')

# Rotate the image
rotated_image = mv.rotate_image(image, angle=45)
mv.save_image(rotated_image, '../mavrick/resources/sample_images/rotated_test.jpg')
