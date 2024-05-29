# mavrick/__init__.py
from .core.image_io import load_image, save_image
from .filters import apply_filter
from .core import load_image, save_image
from .filters import gaussian_filter

__all__ = ['load_image', 'save_image', 'gaussian_filter']
