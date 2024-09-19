# API Reference

## `mavrick.core.image_io`

### `load_image(path)`

- **Description**: Loads an image from the given path and returns it as a NumPy array.
- **Arguments**:
  - `path` (str): Path to the image file.
- **Returns**: `numpy.ndarray`

### `save_image(image_array, path)`

- **Description**: Saves a NumPy array as an image to the given path.
- **Arguments**:
  - `image_array` (numpy.ndarray): Image data to save.
  - `path` (str): Path to save the image file.

## `mavrick.filters`

### `apply_filter(image, filter_matrix)`

- **Description**: Applies a filter to an image using JAX for computations.
- **Arguments**:
  - `image` (numpy.ndarray): The input image as a NumPy array.
  - `filter_matrix` (numpy.ndarray): The filter to apply.
- **Returns**: `numpy.ndarray`
