# mavrick/tests/test_image_io.py
import unittest
from mavrick.core.image_io import load_image, save_image
import numpy as np

class TestImageIO(unittest.TestCase):
    def test_load_image(self):
        image = load_image('mavrick/resources/sample_images/test.jpg')
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (128, 128, 3))  # Assuming a 128x128 RGB image

    def test_save_image(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        save_image(image, 'mavrick/resources/sample_images/output_test.jpg')
        saved_image = load_image('mavrick/resources/sample_images/output_test.jpg')
        np.testing.assert_array_equal(image, saved_image)

if __name__ == '__main__':
    unittest.main()
