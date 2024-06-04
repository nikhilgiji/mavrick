import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import convolve2d


def grayscale_filter(image):
    """Converts an RGB image to grayscale."""
    return jnp.dot(image[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))

# JIT-compile the function
grayscale_filter = jit(grayscale_filter)

def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
    x = jnp.linspace(-size // 2, size // 2, size)
    gauss = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel = jnp.outer(gauss, gauss)
    kernel /= jnp.sum(kernel)
    return kernel

def gaussian_filter(image, size=5, sigma=1.0):
    """Applies a Gaussian filter to an image using JAX for computations."""
    kernel = gaussian_kernel(size, sigma)
    if image.ndim == 2:  # Grayscale image
        return jnp.clip(convolve2d(image, kernel, mode='same'), 0, 255)
    elif image.ndim == 3:  # RGB image
        channels = [convolve2d(image[..., i], kernel, mode='same') for i in range(image.shape[-1])]
        return jnp.clip(jnp.stack(channels, axis=-1), 0, 255)
    else:
        raise ValueError("Unsupported image dimensions.")

# JIT-compile the main function with size and sigma as static arguments
gaussian_filter = jit(gaussian_filter, static_argnums=(1, 2)) 

def brightness_filter(image, factor):
    """Adjusts the brightness of an image by scaling pixel values."""
    return jnp.clip(image * factor, 0, 255)

# JIT-compile the function
brightness_filter = jit(brightness_filter, static_argnums=(1,))

def contrast_filter(image, factor):
    """
    Adjusts the contrast of an image.

    Parameters:
    image (jnp.ndarray): Input image.
    factor (float): Contrast adjustment factor. 
                    1.0 means no change, less than 1.0 reduces contrast,
                    greater than 1.0 increases contrast.

    Returns:
    jnp.ndarray: Image with adjusted contrast.
    """ 
    mean = jnp.mean(image, axis=(0, 1), keepdims=True)
    return jnp.clip((image - mean) * factor + mean, 0, 255)

