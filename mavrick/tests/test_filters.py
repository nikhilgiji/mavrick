import jax.numpy as jnp
import numpy as np
from mavrick.filters import gaussian_filter, grayscale_filter, brightness_filter, contrast_filter

def test_gaussian_filter():
    # Create a simple test image (e.g., a 3x3 grayscale image)
    test_image = jnp.array([[10, 20, 10],
                            [20, 40, 20],
                            [10, 20, 10]], dtype=jnp.float32)

    # Apply Gaussian filter
    filtered_image = gaussian_filter(test_image, size=3, sigma=1.0)

    # Check the shape of the output image
    assert filtered_image.shape == test_image.shape, "Gaussian filter output shape mismatch"

    # Check if the output is still a valid image (e.g., values between 0 and 255)
    assert jnp.all(filtered_image >= 0) and jnp.all(filtered_image <= 255), "Gaussian filter output values out of range"

def test_grayscale_filter():
    # Create a simple RGB test image (e.g., 2x2 image with 3 channels)
    test_image = jnp.array([[[100, 150, 200], [50, 50, 50]],
                            [[200, 50, 100], [25, 75, 125]]], dtype=jnp.float32)

    # Apply Grayscale filter
    grayscale_image = grayscale_filter(test_image)

    # Check the shape of the output image (should be 2x2)
    assert grayscale_image.shape == (2, 2), "Grayscale filter output shape mismatch"

    # Check if the output is still a valid image (e.g., values between 0 and 255)
    assert jnp.all(grayscale_image >= 0) and jnp.all(grayscale_image <= 255), "Grayscale filter output values out of range"

def test_brightness_filter():
    # Create a simple test image (e.g., a 3x3 grayscale image)
    test_image = jnp.array([[10, 20, 10],
                            [20, 40, 20],
                            [10, 20, 10]], dtype=jnp.float32)

    # Apply Brightness filter
    bright_image = brightness_filter(test_image, factor=1.5)

    # Check the shape of the output image
    assert bright_image.shape == test_image.shape, "Brightness filter output shape mismatch"

    # Check if the output is still a valid image (e.g., values between 0 and 255)
    assert jnp.all(bright_image >= 0) and jnp.all(bright_image <= 255), "Brightness filter output values out of range"

def test_contrast_filter():
    # Create a simple test image (e.g., a 3x3 grayscale image)
    test_image = jnp.array([[10, 20, 10],
                            [20, 40, 20],
                            [10, 20, 10]], dtype=jnp.float32)

    # Apply Contrast filter
    contrast_image = contrast_filter(test_image, factor=1.5)

    # Check the shape of the output image
    assert contrast_image.shape == test_image.shape, "Contrast filter output shape mismatch"

    # Check if the output is still a valid image (e.g., values between 0 and 255)
    assert jnp.all(contrast_image >= 0) and jnp.all(contrast_image <= 255), "Contrast filter output values out of range"

if __name__ == "__main__":
    test_gaussian_filter()
    test_grayscale_filter()
    test_brightness_filter()
    test_contrast_filter()
    print("All tests passed!")
