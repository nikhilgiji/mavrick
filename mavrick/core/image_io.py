# mavrick/core/image_io.py
from PIL import Image
import numpy as np
import jax.numpy as jnp 
from jax import jit

def load_image(path):
    """Loads an image from the given path and returns it as a NumPy array."""
    image = Image.open(path)
    return np.array(image)

def save_image(image_array, path):
    """Saves a NumPy array as an image to the given path."""
    # Convert JAX array to NumPy array if needed
    if isinstance(image_array, jnp.ndarray):
        image_array = np.array(image_array)
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(path)

@jit
def resize_image(image, new_size):
    """Resizes an image to the specified size (width, height)."""
    pil_image = Image.fromarray(np.uint8(image))
    resized_image = pil_image.resize(new_size, Image.ANTIALIAS)
    return jnp.array(resized_image)

@jit
def crop_image(image, top, left, height, width):
    """Crops an image to the specified dimensions (top, left, height, width)."""
    pil_image = Image.fromarray(np.uint8(image))
    cropped_image = pil_image.crop((left, top, left + width, top + height))
    return jnp.array(cropped_image)

@jit
def rotate_image(image, angle):
    """Rotates an image by the specified angle."""
    pil_image = Image.fromarray(np.uint8(image))
    rotated_image = pil_image.rotate(angle)
    return jnp.array(rotated_image)

@jit 
def flip_horizonal(image)