import matplotlib.pyplot as plt
import jax.numpy as jnp

class ImageProcessor:
    @staticmethod
    def decode_image_grayscale(image_bytes, width, height):
        # Decode raw grayscale image bytes (example implementation)
        image_size = width * height
        image_array = [image_bytes[i] for i in range(image_size)]
        return image_array

    @staticmethod
    def decode_image_rgb(image_bytes, width, height):
        try:
            # Assuming the image_bytes represent RGB pixel values (3 bytes per pixel: R, G, B)
            image_array = jnp.array(image_bytes).reshape((height, width, 3))
            red_channel = image_array[:, :, 0]
            green_channel = image_array[:, :, 1]
            blue_channel = image_array[:, :, 2]
            return red_channel, green_channel, blue_channel
        except Exception as e:
            print(f"Error decoding RGB image: {e}")
            return None

    @staticmethod
    def imread(file_path, color='rgb'):
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()

            if color == 'rgb':
                red_channel, green_channel, blue_channel = ImageProcessor.decode_image_rgb(image_bytes, width, height)
                # Further processing for RGB image
                return red_channel, green_channel, blue_channel
            elif color == 'grayscale':
                image_array = ImageProcessor.decode_image_grayscale(image_bytes, width, height)
                # Further processing for grayscale image
                return image_array
            else:
                raise ValueError("Invalid color mode. Use 'rgb' or 'grayscale'.")
        except Exception as e:
            print(f"Error reading the image: {e}")
            return None

    @staticmethod
    def imshow(image):
        try:
            if isinstance(image, tuple):  # RGB image format
                red_channel, green_channel, blue_channel = image
                image_data = jnp.dstack((red_channel, green_channel, blue_channel))
            else:  # Grayscale image format
                image_data = jnp.array(image)

            plt.imshow(image_data.astype('uint8'))
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error displaying the image: {e}") 
