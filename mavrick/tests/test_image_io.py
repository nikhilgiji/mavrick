import unittest
from core.image_io import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # Sample grayscale image data (replace this with your actual test data)
        self.grayscale_bytes = bytearray([i for i in range(100)])

        # Sample RGB image data (replace this with your actual test data)
        self.rgb_bytes = bytearray([i for i in range(200)])

        self.width = 10  # Replace with the actual image width
        self.height = 10  # Replace with the actual image height

    def test_decode_image_grayscale(self):
        decoded_image = ImageProcessor.decode_image_grayscale(self.grayscale_bytes, self.width, self.height)
        self.assertIsNotNone(decoded_image)
        self.assertEqual(decoded_image.shape, (self.height, self.width))

    def test_decode_image_rgb(self):
        red_channel, green_channel, blue_channel = ImageProcessor.decode_image_rgb(self.rgb_bytes, self.width, self.height)
        self.assertIsNotNone(red_channel)
        self.assertIsNotNone(green_channel)
        self.assertIsNotNone(blue_channel)
        self.assertEqual(red_channel.shape, (self.height, self.width))
        self.assertEqual(green_channel.shape, (self.height, self.width))
        self.assertEqual(blue_channel.shape, (self.height, self.width))

if __name__ == '__main__':
    unittest.main()
