# mavrick/core/image_io.py
from PIL import Image
import numpy as np
import jax.numpy as jnp 

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
