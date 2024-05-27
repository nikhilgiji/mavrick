# mavrick/filters/__init__.py
import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import convolve2d 
from .gaussian_filter import apply_gaussian_filter


@jit
def apply_filter(image, filter_matrix):
    """Applies a filter to an image using JAX for computations.
    
    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        filter_matrix (numpy.ndarray): The filter to apply.
        
    Returns:
        numpy.ndarray: The filtered image.
    """
    # Ensure the image is 2D or 3D (grayscale or RGB)
    if image.ndim == 2:  # Grayscale image
        return jnp.clip(convolve2d(image, filter_matrix, mode='same'), 0, 255)
    elif image.ndim == 3:  # RGB image
        # Apply the filter to each channel separately
        channels = [convolve2d(image[..., i], filter_matrix, mode='same') for i in range(image.shape[-1])]
        return jnp.clip(jnp.stack(channels, axis=-1), 0, 255)
    else:
        raise ValueError("Unsupported image dimensions.")
